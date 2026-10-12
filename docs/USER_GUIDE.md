# User guide (for people who are not GIS or satellite experts)

This guide explains the app in everyday words. You do not need any programming knowledge.

## What the app does, in one paragraph

It takes two satellite photos of the same place, taken about a year apart, and looks for
ground that was bare or cleared in the newer photo but was green in the older one. Those
spots are the likely new mines. For each spot it checks the official mining permit map
(the **concessions**) and says whether the spot is inside a permit, close to a permit whose
status is unclear, or outside all permits. The result is a map, a list, and a report.

## Before you start

- Run **`scripts\setup.ps1`** once, then **`scripts\run_app.bat`** each time you want to use the app.
- Have a concessions file if you can get one. Without it, every site is marked **illegal**.

## Step by step

1. **Home.** Click **Try the demo** to see an example. The numbers are invented, so the demo
   is for learning the screens only.
2. **Analyze my area.**
   - *Step 1, area:* draw a box on the map, type four numbers, or pick an example. Keep the area
     small. The page tells you its size in km².
   - *Step 2, dates:* the default compares the same date one year apart. That keeps the season
     the same, so ordinary seasonal changes are less likely to be counted as mining.
   - *Step 3, concessions:* upload the permit map if you have it.
   - *Step 4, run:* the progress bar shows what the tool is doing. It can take several minutes.
3. **Results.**
   - The **map** shows each alert. Red is illegal, orange is disputed, green is legal. Click a
     shape for its confidence and area. Use the layer button to switch to satellite view.
   - The **cards** give the total number of alerts and hectares.
   - The **table** can be sorted. Use the filters to hide some statuses or low-confidence alerts.
   - The **downloads** give you the GeoJSON (for GIS software), a CSV (for spreadsheets), and a
     one-page HTML report (open it in a browser and print to PDF).
   - The **caveats** box is always shown. Read it before you use the numbers.
4. **Review.** If the detector was unsure about some places, you can look at small pictures and
   label them. Your labels help check how well the detector works.
5. **Datasets.** Shows which data files are on your computer and lets you fetch missing ones.
6. **Models.** Shows how the deep-learning models scored in tests. Read the honesty panel first.

## What a red, orange or green alert means

- **Illegal (red):** the new disturbance is outside every concession. It is a candidate for an
  unlicensed mine, but check the permit records before acting.
- **Disputed (orange):** the site is in an area whose status is unclear, or it lies close to a
  permit whose status is uncertain.
- **Legal (green):** the site is inside an active concession. Mining there may still break the
  permit's conditions, which the tool cannot see.

**An alert is a lead, not proof.** Roads, quarries, farms and burn scars can look like mining
from space.

## Glossary

**Alert.** One area where the satellite images show new ground disturbance.

**Concession.** An official permit to mine a defined area. Concession maps are usually held by
a government mining registry.

**Confidence.** How sure the model is that a site is new mining, from 0% to 100%. Higher is
more sure. It is not the same as the chance the site is illegal.

**Dice.** A score from 0 to 1 that measures how well a predicted mining area overlaps the true
one. Like IoU, it gives more weight to overlap. 1 is perfect.

**F1.** A score from 0 to 1 that balances missed sites (low recall) against false alarms (low
precision). 1 is perfect. A high F1 on the test set does not guarantee good results elsewhere.

**Full / lite model.** The lite version is smaller and faster but usually a little less accurate.

**Ha (hectare).** 10,000 square metres, about the size of a football pitch and a half.

**IoU (intersection over union).** The share of the combined predicted and true mining area that
both cover. Also called the Jaccard index. 1 is perfect.

**NDVI (normalised difference vegetation index).** A number from about -1 to 1 that is high where
the ground is dense green vegetation and low where it is bare. Mining usually lowers it.

**NDMI (normalised difference moisture index).** A number that rises with moisture in plants and
soil. It helps separate wet ground and water from bare, dug-up ground.

**NDWI (normalised difference water index).** A number that is high over open water. It helps
the detector ignore rivers and lakes.

**BSI (bare soil index).** A number that is high where the ground is bare soil. Dug-up mining
ground often scores high on it.

**Overfitting.** When a model memorises its training data and does poorly on new data. The
Results page flags it with `overfit_flag`.

**Sentinel-2.** A pair of European Space Agency satellites that take colour and infrared images
of the whole Earth about every five days. Images are free to use.

**SMOTE.** A method that creates extra synthetic examples of the rare class (mining) so a model
does not just learn to say "no mining" every time. It is applied only to training data.

**Train / test split.** Part of the data is used to teach the model, and the rest is kept back
to check it. The split here is by area, so the test areas are different from the training areas.

**Caveat.** A limit on how the numbers should be read. Always read the caveats before you use
an alert.

**t0 and t1.** The earlier and later dates being compared.
