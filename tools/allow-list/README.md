# CID evaluations for allow list creation

## Introduction

The manual CID evaluation was an odd idea that turned out to work well enough to keep. If I’d realized I would have to explain it I probably wouldn’t have tried it. Now I publish all this in the hope rather than the expectation that others will find it useful.

Each CID is rated under three rubrics, listed below. The scores are 0 for worst possible quality under a given rubric and 4 for best possible. The scores are subjective and poorly distributed; they still turned out to be useful. (See the last section of this page for some further reflections.) There’s also a notes field where I put things I knew I might want to find later.

This was done mostly by parsing the constituent CID names out of ARD metadata and assembling VRTs that I’d look at in QGIS. This had various small problems, such as VRTs not working with multiple projections, so CIDs on UTM boundaries got split.


## Rubrics (columns or dimensions)

### Water

Percentages are loose guidelines. I only referred to them when I was feeling stumped.

0. Mostly water.
1. Large amounts of water (20 to 50% of surface).
2. Water is obvious at thumbnail scale (5 to 20%).
3. Large rivers, medium reservoirs, or small lakes (1 to 5%). Typical inland in wet to moderate climates.
4. Only small waterways, agricultural ponds, home pools, and similar (< 1%). Typical inland in dry climates.

### LULC complexity

This dimension combines diversity of landcover with human influence on landcover: basically, is the visible land surface going to be relatively valuable for training a pansharpener that will be used mainly on cities?

0. Zero to very rare buildings, clearings, farms, and other clear human traces; monotonous or barren landcover.
1. Hamlets and local primary industry; unremarkable landcover. Rural areas, most forests, and mechanized farms.
2. Villages and towns, or more than half the land area is obviously human-influenced; interesting landcover. Exurbs and farmland, or some especially interesting natural landscapes.
3. At least moderate urbanization or locally intensive agriculture/industry; richly interesting landcover. Suburbs and midsized cities.
4. City cores and large industrial zones; XKCD 1472. Sometimes I might promote a 3 with especially good vibes.

### Seeing

This is in [the astronomical sense](https://en.wikipedia.org/wiki/Astronomical_seeing): how well can we see things? Here it’s almost entirely about clouds. The levels are:

0. No ground is fully visible, or virtually none. Rejected at first glance.
1. Severe problems, but some areas could plausibly be useful for something.
2. Many clouds but also substantial clear areas. Would be reasonably useful given perfect cloud detection.
3. Some visible clouds that would be easy to manually cut out.
4. Perfect up to tiny wisps of fog and minor anthropogenic clouds like vapor from power stations.

Seeing is the only dimension where I sometimes used fractional numbers. CIDs that were nearly but not quite cloud-free would occasionally get a 3.75 or something.

## Formulas

Depending on your interests, desire to avoid water and clouds, and so on, you will of course want to make your own weighting formula. The dimensions are designed so that you can add, average, or multiply them, with positive weights, and select the highest scores as the best images. It may also be helpful to add nonlinearity, because there’s no absolute scale defined here; in some dimension, for some purpose, you may consider a 2 much more or much less than half as good as a 4.

The one big gotcha is to make sure you’re filtering for WV-{2,3} satellites (assuming that’s what you want). Their CIDs start with 103 and 104, which looks like a typo but is not. My impression from public information as of 2025-02-03 is that the Legion series will have a very similar sensor to the WV-{2,3} generation and its CIDs will start with 2, but I could easily have misunderstood or missed something.

A formula that I’ve used is the product $w\times l\times s$ of three adjusted versions of the rating rubrics,

```math
\begin{gathered}
w = \sqrt{\frac{\mathrm{water}}{4}} \\
l = \left(\frac{\mathrm{LULC}}{4}\right)^2 \\
s = max(0, \mathrm{seeing} - 3)
\end{gathered}
```

All three values are thus scaled into the unit range: water and LULC by division, seeing by shifting it so that only values that started in the range 3..4 are counted. Water and LULC are also nonlinearly scaled, but oppositely: the square root on water makes it superlinear, because a rating of say 3 is still good for my purposes; LULC selects more strictly, as ratings of about 2 and under add very little to the mix. Because this is a product, seeing (or rather _s_) ending up at 0 for most CIDs means that most CIDs have a score of 0.


## Using the CSV (applying formulas to the ratings)

The ratings are delivered as a CSV that should be easy to import to your database, dataframe notebook, or spreadsheet app. On the CLI, I’ve heard good things about [csvkit](https://csvkit.readthedocs.io/en/latest/), but here I’ll use [duckdb](https://duckdb.org/):

```sql
-- for clarity, we load the CSV as a table
create table ratings as select(*) from read_csv('ratings.csv', auto_detect=true);

-- I used a free SQL formatting service but it
-- came out all messed up like this ¯\_(ツ)_/¯
with ranked as
  (select cid,
     -- see section above for commentary on this particular formula
     pow(water/4, 0.5) * pow(lulc/4, 2) * greatest(0, (seeing - 3)) as score
   from ratings)
select cid,
       round(score, 3)
from ranked
where cid[:3] in ('103',
                  '104')
order by score desc
limit 10;
```

This shows us the top-ranked CIDs (which score 1, because there are some CIDs that have 4 water, 4 LULC, and 4 seeing, which with scaling is 1×1×1). We could also do something like this:

```sql
copy 
  (with ranked as
    (select cid,
            pow(water/4, 0.5) * pow(lulc/4, 2) * greatest(0, (seeing - 3)) as score
     from ratings)
  select cid
  from ranked
  where cid[:3] in ('103', '104')
    and score > 0.225
  order by score desc)
to 'new-allow-list.txt' (header false);
```

We can use this new file with `train.py`. The 0.225 cutoff is ad-hoc and not special. You could also use a `limit`, for example, although I’d be surprised if you were chipping every single scene in the Open Data Program, so it probably only makes sense if you’re also selecting for the `event`s you have at hand.


## Reflections and regrets

Given a time machine, here’s what I would tell myself before doing this:

- CIDs are coarser than the idea being applied; they aren’t the right unit of analysis. There are several really beautiful collects disqualified by a few big clouds in one corner (this is basically what seeing=3 means). The best way is probably something like a very simple polygon-drawing tool to make an “allow polygon” attached to each CID, such that ARD tiles that fall entirely within the polygon are accepted. Maybe the output CSV is a list of quadkeys or maybe it’s CIDs and geojson polygons. Maybe it makes more sense to do deny polygons instead, or maybe the chipper should actually use the polygons to do in-tile masking. Any of these would certainly add complexity, but I think they could probably increase the useful information by something like 1/3 on the same data.

- I imagine my ratings drifted over time; for example, I think I got less strict about what could go in water=4 as I went along. I never felt like I knew whether I wanted to call a LULC 4, partly because some CIDs have some really dense urban fabric and then also cattle fields at the other end – see previous point. I don’t regret cutting corners on proper methodology for human ratings in this case simply because rating a thousand large images was hard enough and I’d rather do it badly than not do it.

- The landcover dimension should probably be two dimensions, one for landcover complexity or rarity and the other for visible human influence. This is tricky; they’re entangled ideas. Possibly a better way to slice it is a landcover dimension of some kind and a personal preference dimension that’s a completely subjective weighting.

- There is room for much more sophistication here. The question is only whether it’s worth the effort. For example, we could record which chips have the highest loss and weight more toward their CIDs (or quadkeys); we could use Maxar’s metadata – or standard indexes or a separate neural network – to rate things more objectively; we could try to crowdsource a nice cross-checking multiplayer rating system; we could do a lot of things. What makes sense depends on the error budget of the project as a whole (which I have not tried to calculate), on whether anyone else in the world is interested enough in this to work on it, and on whether this particular dataset will continue as the best available.
