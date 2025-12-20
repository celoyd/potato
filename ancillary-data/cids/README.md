# CID evaluations for allow list creation

Here we filter useful training images out of the Maxar Open Data Program dataset. The techniques presented may be useful in thinking about other datasets as well.

A CID is a Maxar _catalog ID_, which names a single collect (i.e., one single contiguous image as seen by the satellite, but generally multiple tiled images as delivered in ARD). A CID looks like `10300100DF069700`.

This directory contains:

- [ratings.csv](ratings.csv), a CSV containing every CID in the Maxar Open Data Program and its subjective rating (I’m the subject) on several quality axes described below. You can use it (with techniques also given below) to replicate:
- [allow-list.csv](allow-list.csv), about 200 of the CIDs that rank highest in a combined quality metric. The CSV filename suffix is only by courtesy; it’s simply one CID per line. This can be used as the argument for `tools/chipping/chip.py --allow-list`.

## Introduction

The manual CID evaluation was an odd idea that turned out to work well enough to keep. If I’d realized I would have to explain it, I might not have tried it.

Each CID is rated under three rubrics, listed below. The scores are 0 for worst possible quality under a given rubric through 4 for best possible. The scores are subjective and poorly distributed; they still turned out to be useful. (See the last section of this page for some further reflections.) There’s also a free-text notes field.

This was done mostly by parsing the constituent CID names out of ARD metadata and assembling VRTs that I’d look at in QGIS. This had various small problems: for example, CIDs on UTM boundaries got split.


## Rubrics (columns or dimensions)

### Water

The percentages at the end are the rounded proportion of WV-{2,3} CIDs that I gave the rating.

0. Mostly water. (I gave ~5% of WV-{2,3} CIDs this rating.)
1. Extensive water – more than about 1/5 of the surface but less than half. (12%.)
2. Water is obvious at thumbnail scale; big lakes, chunks of ocean, and so on. (20%.)
3. Large rivers, medium reservoirs, or small lakes. Typical inland in wet to moderate climates. (42%.)
4. Only small waterways, agricultural ponds, home pools, and similar. Typical inland in dry climates. (22%.)

### LULC complexity

This dimension combines diversity of landcover with intensity of human influence on landcover: basically, is the visible land surface going to be relatively valuable for training a general-purpose pansharpener?

0. Zero to very rare buildings, clearings, farms, and other clear human traces; monotonous or barren landcover. (I gave ~2% of WV-{2,3} CIDs this rating.)
1. Hamlets and local primary industry; unremarkable landcover. Rural areas, most forests, and light agricultural regions. (16%.)
2. Villages and towns, or more than half the land area is obviously human-influenced; interesting landcover. Exurbs and farmland, and some especially interesting natural landscapes. (40%.)
3. At least moderate urbanization or locally intensive agriculture/industry; richly interesting landcover. Suburbs and midsized cities. (25%.)
4. City cores and large industrial zones; XKCD 1472. Sometimes I might promote a 3 with especially good vibes. (18%.)

### Seeing

This is in [the astronomical sense](https://en.wikipedia.org/wiki/Astronomical_seeing): how well can we see things? This is important because clouds show heavy band separation, so it’s not just that they aren’t interesting – to some degree they actually poison the data. The levels are:

0. No ground is clearly visible, or virtually none. Rejected at first glance. (I gave ~1% of WV-{2,3} CIDs this rating.)
1. Severe problems, but some areas contain recognizable ground detail; might be useful for some other project. (4%.)
2. Many clouds but also substantial clear or only lightly hazy areas. Could be useful given perfect cloud masks. (21%.)
3. Some visible clouds that would be easy to manually cut out; remaining areas are near-perfect. (29%.)
4. Perfect up to tiny wisps of fog and minor anthropogenic clouds like vapor from power stations. (45%.)

Seeing is the only dimension where I sometimes used fractional numbers. CIDs that were nearly but not quite cloud-free would occasionally get a 3.75 or something.

## Formulas

Depending on your interests, desire to avoid water and clouds, and so on, you will of course want to make your own weighting formula. The ratings are designed so that you can add, average, or multiply them, with positive weights, and select the highest scores as the best images. It may also be helpful to add nonlinearity, because there’s no absolute scale defined here; in some dimension, for some purpose, you may consider a 2 much more or much less than half as good as a 4.

The one big gotcha is to make sure you’re filtering for WV-{2,3} satellites (assuming that’s what you want). Their CIDs start with 103 and 104, which looks like a typo but is not.

A formula that I’ve used is the product $w \times l \times s$ of three adjusted versions of the rating rubrics,

```math
\begin{align*}
w &= \left(\frac{\mathrm{water}}{4}\right)^{1/2} \\
l &= \left(\frac{\mathrm{LULC}}{4}\right)^2 \\
s &= max(0, \mathrm{seeing} - 3)
\end{align*}
```

All three values are thus scaled into the unit range: water and LULC by division, seeing by counting only its level over 3. Water and LULC are also nonlinearly scaled, but oppositely: the square root on water makes it superlinear, because a rating of say 3 is still good for my purposes; LULC selects more strictly, as ratings of about 2 or under add little useful information to the mix. Because the score is a product, seeing (or rather $s$) ending up at 0 for most CIDs means that most CIDs have a score of 0. In other words, a marvelous image will be taken out of consideration if it’s cloud-contaminated, which is what we want.


## Using the CSV (applying formulas to the ratings)

The ratings are delivered as a CSV that should be easy to import to your database, dataframe notebook, [csvkit](https://csvkit.readthedocs.io/en/latest/), spreadsheet app, or other tool. Here I’ll use [duckdb](https://duckdb.org/). Run it in this folder and you can do this:

```sql
-- for clarity, we load the CSV as a table
create table cids as select(*) from 
  read_csv(
    'ratings.csv',
    header=true,
    columns = {
      'cid': 'varchar',
      'event': 'varchar',
      'water': 'real',
      'lulc': 'real',
      'seeing': 'real',
      'notes': 'varchar'
    });

-- add a column for ratings (see above for notes on formula)
alter table cids add column score real default 0.0;

update cids
  set score = round(
    pow(water/4, 0.5) * pow(lulc/4, 2) * greatest(0, (seeing - 3)),
    3
  )
  where cid[:3] in ('103', '104');
```

We can now run something like:

```sql
select cid, score from cids order by score desc limit 10;
```

To see the top-rated CIDs. They have score 1 because there are some CIDs that have 4 water, 4 LULC, and 4 seeing, which with scaling is 1×1×1).

Now we can, for example, adjust the score formula (updating its column), and eventually write out a new allow list:

```sql
copy 
  (with ranked as
    (select cid, score from cids)
  select cid
  from ranked
  where score > 0.225
  order by score desc)
to 'new-allow-list.csv' (header false);
```

This new file will work with `chip.py`. The 0.225 cutoff is not special; it’s as specific to my tastes and purposes as the score formula is.


## Reflections and regrets

Given a time machine, here’s what I would tell myself before doing this:

- CIDs are coarser than the idea being applied; they aren’t the right unit of analysis. There are several really beautiful collects disqualified by a few big clouds in one corner (this is basically what seeing=3 means). Is the LULC dimension about the average or the best LULC in the collect? The best way forward is probably something like a very simple polygon-drawing tool to make an _allow polygon_ attached to each CID. Then every ARD tile that falls entirely within the polygon is allowed. Or maybe it’s a deny polygon. Maybe the output is a list of tiles, or maybe the chipper should receive the polygon and do in-tile masking. Any of these would certainly add complexity, but I think they could probably increase the useful information by something like 1/3 on the same data.

- I imagine my ratings drifted over time; for example, I think I got less strict about what could go in water=4 as I went along. I never felt like I knew whether I wanted to call a LULC 4, partly because some CIDs have some really dense urban fabric and then also cattle fields at the other end – see previous point. I don’t regret cutting corners on proper methodology for human ratings in this case simply because rating a thousand large images was hard enough and I’d rather do it badly than not do it.

- The landcover dimension should probably be two dimensions, one for landcover complexity or rarity and the other for visible human influence. This is tricky; they’re entangled ideas. Possibly a better way to slice it is a landcover dimension of some kind and a personal preference dimension that’s a completely subjective per-project weighting. Probably sometimes I was rating by maximum LULC interest and other times by average. A huge strip like 10500100450FC000, which contains a lot of water and moderately interesting rural land (~2) but also Port-au-Prince (4) is an example. In terms of how likely a chip from this strip is to be a pansharpening challenge, it’s middle-of-the-road. In terms of where to look to find difficult chips, it’s spectacular. The problem is to usefully distinguish those ideas.

- There are also imaginable dimensions for noise (related to which sensor it is, to lighting, and to intensity of atmo, I think) and terrain correction artifacts. You wouldn’t want to predict terrain correction from off-nadir angle alone because it depends on the actual terrain, on the quality of the terrain model, and on the quality of the tiedown. 

- Ratings should probably be a 7 point scale; 5 gets a little constricting. Maybe I should just use decimals more.

- There is room for much more sophistication here. The question is only whether it’s worth the effort. For example, we could record which chips have the highest quality and weight more toward their CIDs (or quadkeys); we could use Maxar’s metadata – or standard indexes or a separate neural network – to rate things more objectively; we could try to crowdsource a nice cross-checking multiplayer rating system; we could do a lot of things. What makes sense depends on the error budget of the project as a whole (which I have not tried to calculate), on whether anyone else in the world is interested enough in this to work on it, and on whether this particular dataset will continue as the best available.

