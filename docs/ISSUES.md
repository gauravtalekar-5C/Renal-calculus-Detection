# Linear issues — Renal Calculus Detection & Measurement

**How to use this file:** each block is one issue. Copy the `Title` line into
Linear's title field and everything under `Description` into the body. Linear
renders this markdown as-is. Create parent issues (`RC-1`, `RC-2`, …) first, then
add each sub-issue (`RC-9.1`, `RC-9.2`, …) underneath its parent.

---

# Read this first — plain-English primer

**What the project does.** A patient with suspected kidney stones gets a CT scan.
A radiologist reads it and writes a report. We are building software that reads
the same scan automatically and reports: are there stones in the kidney, how
many, how big, where exactly, and how dense.

**Why anyone wants this.** Stone size and count decide treatment. A 4 mm stone
usually passes on its own; a 10 mm stone usually needs intervention. Counting and
measuring by hand across 200 image slices is slow and inconsistent between
readers, so an automated measurement is genuinely useful — but only if it is
reliable enough to trust.

**Terms used throughout**

| term | what it means |
|---|---|
| **calculus / calculi** | the medical word for stone / stones |
| **CT scan** | a stack of cross-sectional X-ray images. One scan = 100–800 image slices |
| **slice thickness** | how thick each image slice is, in mm. Thin slices show small objects; thick slices blur them away |
| **HU (Hounsfield Unit)** | the density scale in CT. Air is −1000, water is 0, fat about −100, muscle about +50, kidney tissue about +30, **stones are +130 to +1500**, bone is +300 to +1500 |
| **plain vs contrast** | "plain" = no injected dye. "Contrast" = iodine dye injected, which lights up blood vessels and urine. Dye is as bright as a stone, so **stones can only be measured on plain scans** |
| **segmentation** | outlining an organ in 3D so the computer knows which voxels are "kidney" |
| **TotalSegmentator** | a free, published AI model that outlines ~117 body structures on CT. We use it to find the kidneys |
| **voxel** | a 3D pixel |
| **threshold** | "call every voxel above X HU a stone candidate" — the simplest possible detector |
| **calyx / poles** | internal compartments of the kidney. Radiologists report stones as upper pole, mid (interpolar), or lower pole |
| **UVJ / VUJ** | where the ureter joins the bladder. Reports give ureteric stone positions as a distance from here |
| **sensitivity** | of the patients who really have stones, what fraction did we find? Misses hurt this |
| **specificity** | of the patients with no stones, what fraction did we correctly call clear? False alarms hurt this |

**How the software works, in one paragraph.** Pick the right image series out of
the scan (plain, thin slices). Skip children, because the kidney-outlining model
is trained on adults and fails on them. Outline both kidneys. Clean up image
noise. Inside the kidney outline only, find everything brighter than 130 HU.
Group touching bright voxels into 3D objects — each object is one candidate
stone. Throw out the ones that are obviously bone or blood-vessel calcium.
Measure what is left. Draw pictures so a human can check.

**Where the project stands.** Detecting *whether* a kidney contains a stone works
at roughly 80% sensitivity and 90% specificity. **Counting and volume have never
been checked against ground truth** — not because we think they are wrong, but
because radiology reports say "a few tiny calculi" and never "4 stones totalling
112 mm³", so there has been nothing to check against. Fixing that is `RC-9` and
it blocks most of the remaining work.

**Naming note:** the project folder is spelled `kindey_calculus_measurement`
(typo in the original folder name, kept to avoid breaking paths).

---
---

# PARENT RC-1

**Title:** `RC-1 · Download pipeline, and the 34-day limit on our own data`

**Labels:** `part-1` `infrastructure` `done`
**Status:** Done
**Priority:** No priority — record of completed work

**Description:**

**What this was.** Before anything else, we needed CT scans to work with. This
issue covers the tool that pulls them from our internal system and the single
most important operational fact we learned doing it.

**What a "study" is.** One patient's scan, delivered as a zip file containing
several hundred to several thousand individual image files. Our download tool
takes a list of study IDs and fetches each one.

**What was delivered**

- **44 studies downloaded** — 70,643 image files, 14.6 GB total
- Resumable: a partly-downloaded file is saved as `.part`, checked for
  corruption, and only then renamed. So an interrupted download costs nothing.
- A `--iuid` flag for grabbing one specific study on demand.
- Files are named by our internal study ID. **Deliberately not** by the name the
  server suggests, because that filename contains the patient's name, which is
  protected health information and must not end up on disk.

**The cohort we ended up with**

| type of stone described in the report | studies |
|---|---|
| ureteric (stone in the tube from kidney to bladder) | 33 |
| urography (a multi-phase dye study) | 10 |
| renal (stone in the kidney itself) | 1 |

Note the skew: only one study was selected *because* it had a kidney stone. Most
of our kidney stones are incidental findings inside studies collected for
ureteric stones. This matters for `RC-14`.

**⚠️ The key finding: our data source only keeps scans for about 34 days.**

Ask for a study older than that and the request hangs for 5 minutes and then
fails. Everything about the download tool is shaped by this:

- **We first assumed the failures were a load problem** — too many requests at
  once. That was wrong. We proved it by testing consecutive days: 28 June always
  fails, 29 June always succeeds. A clean cut-off by date, not by load. Old scans
  have been moved to slow storage and cannot be retrieved at all.
- **The tool's own timeout must be longer than the server's.** Ours was initially
  shorter, so we saw a generic "connection timed out" instead of the server's
  actual error message — which sent us investigating the wrong thing for a while.
- **Never retry these failures.** An unavailable scan fails identically every
  time. Retrying cost about 19 minutes per study for no benefit.
- A separate small tool (`probe_api.py`) can re-measure today's cut-off date in a
  couple of minutes by starting each download and aborting it after 4 MB.

**What this means for planning.** Any future data collection must happen within
about a month of the scan being taken. We cannot go back and collect a historic
dataset. Nine of our download attempts failed permanently for this reason. If we
want more data, someone has to pull it *soon* — see `RC-14`.

**Code:** `utils/download_dicoms.py`, `utils/build_worklist.py`,
`utils/probe_api.py`

---
---

# PARENT RC-2

**Title:** `RC-2 · Choosing the right images out of each scan`

**Labels:** `part-1` `pipeline` `done`
**Status:** Done

**Description:**

**The problem this solves.** One CT scan is not one set of images. It typically
contains about eight different versions of the same body region — before dye,
after dye, delayed views, thick summary images, and a low-quality positioning
shot. Only one of them is suitable for measuring stones. This stage picks it
automatically.

**What we require, and why each rule exists**

**Must be "plain" (no injected dye).** Iodine dye in the urine measures brighter
than 130 HU, which is the same brightness as a stone. On a dye-enhanced scan a
computer literally cannot tell them apart by brightness — and neither can a
human, which is why radiologists also use the plain images for stones.

**Must be thin slices — 1.5 mm or less.** This is the rule people find
surprising, so here is the reason. A CT image slice is an average over its whole
thickness. Put a 2 mm stone in a 3 mm-thick slice and its brightness gets
averaged with the surrounding kidney tissue, so it reads far dimmer than it
really is. We tested this on a computer-generated stone of known size and
brightness: **a 2 mm stone at 300 HU disappears completely at 3 mm slice
thickness.** It is not "harder to find" — there is nothing left to find. This
matches the published recommendation (Kambadakone et al.) and is why 4 of our 44
studies had to be discarded.

**Must cover enough of the body** (at least 250 mm and 40 slices) so we are not
looking at a partial scan that cuts off half a kidney.

**Must be axial** (the standard cross-sectional orientation).

**How we tell plain from dye-enhanced.** Five methods, tried in order, each
falling through to the next if it cannot decide: the radiographer's own text
label saying plain → the label saying contrast → a header field → the brightness
of the aorta compared to the spine → the brightness of the aorta on its own.
The last one is used last because it is the least reliable: aorta brightness
varies with timing, so on its own it produces wrong answers.

**Result across our 44 studies**

| verdict | studies | meaning |
|---|---|---|
| usable | **37** | plain, thin, full coverage |
| detectable but not measurable | 2 | 2–3 mm slices — can spot a stone, sizes unreliable |
| too thick | 2 | 5 mm slices — small stones invisible |
| unusable | 3 | no suitable images at all |

Of the 37 usable, 14 are finer than 0.75 mm and 23 sit in the 0.75–1.5 mm band.

**Code:** `utils/triage_series.py`

---
---

# PARENT RC-3

**Title:** `RC-3 · Excluding children, because the kidney-outlining model fails on them`

**Labels:** `part-1` `pipeline` `done`
**Status:** Done

**Description:**

**The rule.** Skip any patient aged 18 or under. Their age is already recorded in
the scan file, so this costs nothing to check.

**Why — and this was discovered, not anticipated.**

We use a free published AI model called TotalSegmentator to outline the kidneys.
It was trained almost entirely on adult scans. On children it does not fail
loudly — it produces an outline that looks like an answer but is badly wrong.

A healthy adult kidney is about 90–220 mL. Here is what we measured:

| patient | body width at kidney level | kidney volume the model produced | verdict |
|---|---|---|---|
| 7-year-old boy | 199 cm² | 97 mL | ❌ far too small |
| 7-year-old boy | 217 cm² | 35 mL | ❌ far too small |
| 18-year-old, small build | 221 cm² | 31 mL, **in 3 disconnected pieces** | ❌ nonsense |
| 34 adults | 298 – 841 cm² | all plausible | ✅ |

Every patient with a small body failed. Every patient with an adult-sized body
succeeded. No exceptions in either direction.

**The 31 mL case is worth understanding**, because it shows why this matters.
That scan produced *five detected stones*, including one reported as 20.3 mm —
which would be a clinically significant finding. All five were meaningless,
because the "kidney" the software searched was three fragments of unrelated
tissue. Looking at the picture makes it obvious: the outline is a stack of
horizontal bands lying on muscle beside the spine. No kidney looks like that.
**Without the picture, the numbers looked perfectly credible.**

**Can we not just use a children's model?** No. We checked all 53 models
TotalSegmentator offers — there is no paediatric one. So these scans cannot be
rescued by swapping models. They have to be excluded, and excluded visibly rather
than quietly reported as 30 mL kidneys full of stones.

**Why we exclude on age rather than body size.** Age catches all three failures
and is simpler. We initially used both criteria and it backfired: a 17-year-old
with a fully adult-sized body and a perfectly normal 232 mL kidney got thrown out
for no reason. We still *measure* body size, because it explains why the model
fails, but it no longer decides anything.

**Two scans have no age recorded** (one contains a placeholder value of "0
years"). Those are kept and flagged rather than dropped — one of them is a normal
adult with healthy 319 mL kidneys, and discarding real data to guard against a
hypothetical is the wrong trade.

**Where this runs.** Before the kidney outlining, not after. Outlining takes
about 2 minutes of expensive GPU time per scan, so there is no point spending it
on a scan we are going to discard.

**Code:** `utils/patient_gate.py`

---
---

# PARENT RC-4

**Title:** `RC-4 · Converting scans to a usable format — and the left/right bug`

**Labels:** `part-1` `pipeline` `done` `postmortem`
**Status:** Done

**Description:**

**What this stage does.** CT scans arrive as hundreds of separate files, one per
image slice. This stage stacks them into a single 3D volume that the rest of the
software can work with, and records how that volume is oriented in the patient's
body — which way is up, which way is forward, and **which side is the patient's
left**.

That orientation information is the tricky part, and we got it wrong.

**⚠️ Postmortem: for several weeks the software had the patient's left and right
swapped.**

**What it meant in practice.** Every "left kidney" was actually the right one.
Every reported side was mirrored. If this had reached a radiologist, we would have
been telling them a stone was in the wrong kidney.

**Why nothing caught it.** This is the important part for anyone building similar
software. The orientation was *recorded* correctly in the file header — the
standard check for this returns the expected answer, so it passed. No test
failed. No error appeared. Every number in every output file looked entirely
reasonable. There was no signal at all.

**How it was actually found.** By looking at a picture. In a CT image displayed
the standard medical way, the patient's right side appears on the **left** of the
screen. The liver sits on the patient's right, so the liver should appear
screen-left — and it did. But the outline we had labelled "left kidney" was drawn
on that same side, next to the liver. Two things that cannot both be true.

**The rule we now follow, and which belongs in team onboarding:**

> Never trust the orientation recorded in the file. Render one picture and check
> that the liver is on the patient's right.

Fixing it meant deleting and regenerating every converted scan, every kidney
outline, and every result produced up to that point.

**Code:** `utils/extract_series.py`

---
---

# PARENT RC-5

**Title:** `RC-5 · Outlining the kidneys, checking the outlines, and drawing them`

**Labels:** `part-1` `pipeline` `done`
**Status:** Done

**Description:**

**What this stage does.** Runs TotalSegmentator to outline the kidneys in 3D,
plus 12 other structures we need for later steps (bladder, aorta, major veins and
arteries, and some bones — these are used to rule out things that look like
stones but aren't). About 2 minutes per scan on our GPU.

**Why this gets its own verification step.** Everything downstream depends on
these outlines. If the kidney outline is wrong, every stone count, size and
location built on top of it is wrong too — and, as `RC-3` and `RC-4` both showed,
wrong in a way that produces believable-looking numbers. So the outlines are
checked on their own, before any stone result is taken seriously.

**Automatic checks.** For each kidney we compute volume, length and average
tissue brightness, and flag anything outside the normal range:

| measurement | normal range |
|---|---|
| volume | 90 – 220 mL per kidney |
| length | 80 – 140 mm |
| tissue brightness | 15 – 55 HU |
| difference between the two sides | less than 2× |

**Pictures.** Numbers alone cannot distinguish a genuinely shrunken kidney from a
failed outline — both give a small volume. So we generate five images per scan,
each answering a different question:

| image | the question it answers |
|---|---|
| `coronal.png` | Are both kidneys found, roughly the right shape, in the right place? **Look at this one first — it catches most failures in a second.** |
| `sagittal.png` | Are the top and bottom of the kidney included, or cut off? Matters because a cut-off bottom means a missed lower-pole stone. |
| `boundary.png` | Outline only, no fill. Does the edge follow the real kidney surface, or has it spilled into the liver or bowel? A filled outline visually hides spilling; an outline-only view doesn't. |
| `axial_grid.png` | 12 cross-sections, chosen at the top and bottom plus wherever the outline changes size most abruptly — i.e. where outlining tends to break, not just evenly spaced. |
| `axial_grid_even.png` | 12 evenly spaced cross-sections, for an unbiased overview. |

Left kidney is drawn blue, right kidney green, in standard medical orientation
(patient's right appears screen-left).

**This step is what caught both major bugs so far** — the child-patient failures
and the left/right swap. Neither was visible in any number.

**Code:** `utils/run_anatomy.py`, `utils/kidney_qc.py`,
`experiment_1/render_kidney_masks.py`

---
---

# PARENT RC-6

**Title:** `RC-6 · Finding and measuring the stones`

**Labels:** `part-1` `pipeline` `done`
**Status:** Done

**Description:**

**The core of the software.** Given a scan and an outline of the kidneys, find
every stone and measure it. We follow a published method (Elton et al. 2022,
whose code is public) and add several safeguards of our own.

**How it works, step by step**

1. **Define where to look.** Search inside the kidney outline, plus the hollow
   central area where stones collect, plus a 3 mm margin.
2. **Clean up image noise.** Random speckle in a CT image can cross the
   brightness threshold and look like a tiny stone.
3. **Find bright things.** Anything above 130 HU is a possible stone.
4. **Group them into objects.** Touching bright voxels become one 3D object.
   This is how counting works, and it must be done in 3D — see below.
5. **Throw out the impostors.** Bone edges and calcium in blood vessels are just
   as bright as stones.
6. **Measure what's left** — size, volume, brightness, and which part of the
   kidney it's in.

**The design decisions, each caused by a specific failure**

**We search a "filled-in" kidney shape, not an expanded one.** Stones collect in
the hollow middle of the kidney, which is not inside the outlined tissue. Our
first attempt simply expanded the outline by 12 mm in all directions — which
reached the middle, but also reached out into fat, bowel and ribs, and started
reporting stones outside the kidney entirely. The fix is to expand and then
shrink back, which fills hollows without growing outward.

**We require one genuinely bright voxel, not just above-130 voxels.** Using a
plain 130 HU threshold, the software reported **81 stones per scan**. Almost all
were noise — the typical "stone" peaked at 139 HU, barely over the line. Adding a
requirement that a real stone must contain at least one voxel above 200 HU cut
this to 9 per scan. (This requirement has its own cost — see `RC-10`.)

**We measure size on the original image, not the noise-cleaned one.** Cleaning
noise slightly shrinks small objects. When we measured on the cleaned image, one
scan's three genuine 2 mm stones vanished entirely. So: cleaned image decides
*where* stones are, original image decides *how big* they are.

**We identify bone by size, not by asking the outlining model.** Bone is as
bright as a stone, and rib and spine edges were being reported as 15–45 mm
stones. We only asked TotalSegmentator for a few specific bones, so ribs and
mid-spine were invisible to us. The robust fix: any bright connected object
bigger than 3000 mm³ is bone. Even a large stone is a few hundred mm³, so the
sizes don't overlap.

**We detect dye by looking at kidney tissue, not the aorta.** One scan reported
9 stones that were all dye in the urine. The obvious check — is the aorta bright?
— fails here, because on delayed dye scans the aorta has already cleared while
the urine is still full of dye. Checking the kidney tissue itself catches it. An
earlier version of this check also rejected scans containing a lot of bright
material, which wrongly threw out a patient with a genuinely large stone; that
part was removed.

**Why counting must happen in 3D.** A single image slice cannot count stones. One
stone can appear as two separate bright patches in one slice and be counted
twice; two stones stacked vertically appear as one and be counted once. And a
3 mm stone appears in only about 5 of 150 slices, so picking any single slice
misses most stones entirely. Detection therefore uses **every** slice containing
kidney — typically 100 to 200 of them.

**How measurement works.** A stone's edge is not sharp in a CT image; brightness
fades over a millimetre or two. We define the edge as the point where brightness
is halfway between the stone's peak and the surrounding tissue — a standard
approach — and for the fading edge voxels we count each one partially rather than
all-or-nothing.

**How accurate is the measurement?** We tested it on computer-generated stones of
exactly known size, with realistic blur and noise added:

| image resolution | size error | volume error |
|---|---|---|
| 0.7 mm | +0.12 mm | −2% |
| 0.8 × 0.8 × 1.25 mm | +0.19 mm | 0% |

So the arithmetic is sound to about a fifth of a millimetre. **Important caveat:
this proves the maths is right, not that the software found the right objects in
a real patient.** That is `RC-9`.

**Code:** `utils/detect_stones.py`, `utils/test_measurement.py`

---
---

# PARENT RC-7

**Title:** `RC-7 · Scoring ourselves against the radiologist's report`

**Labels:** `part-1` `validation` `done`
**Status:** Done

**Description:**

**What this measures.** For each scan we already have a radiologist's written
report. This compares what our software found against what the radiologist wrote,
which tells us how good the software is without needing anyone to do new work.

**One important restriction.** Part 1 only looks inside the kidney. Many of our
scans describe a stone in the ureter — the tube from kidney to bladder — which is
outside our search area. For those, finding nothing is the *correct* answer, so
scoring must only count kidney stones. Reports are searched for kidney-specific
words (calyx, pole, renal pelvis, staghorn) rather than just "calculus".

**Where ground truth comes from.** The spreadsheet accompanying our data has a
column recording where each stone was — `renal`, `ureteric`, `VUJ`, or
combinations. We use that as the answer key, and separately run our own text
search over the report as a cross-check. **The two agree on 35 of 37 scans**,
which is reassuring about both.

**Results — 37 scans, 31 of which could be scored**

(6 were excluded: 3 dye-enhanced, 3 paediatric.)

| what we measured | result | realistic range |
|---|---|---|
| **sensitivity** — of scans with a kidney stone, how many did we find? | **79%** (15 of 19) | **57% – 91%** |
| **specificity** — of scans with no kidney stone, how many did we correctly clear? | **92%** (11 of 12) | **65% – 99%** |
| **which side** (left/right), when we did find one | **100%** (15 of 15) | — |

**Please read the ranges, not just the headline percentages.** With only 19
positive scans, "79%" is a single estimate from a small sample — the true figure
could plausibly be anywhere from 57% to 91%. The ranges are that wide because the
sample is small, not because the software is erratic. The same performance
measured on 100 scans would narrow to roughly 70–86% and 85–96%.

**So the honest way to state our position is: "roughly 80% sensitivity and 90%
specificity, on a small sample."** Not "79%".

Improving the *precision of that estimate* now needs more scans (`RC-14`), not
better software.

**What this scoring method fundamentally cannot check.** Reports say *"a few tiny
calculi"*. They do not say "4 stones totalling 112 mm³". So **counting and volume
cannot be validated this way at all** — that needs `RC-9`.

**Code:** `utils/summarize.py`, `utils/compare_reports.py`

---
---

# PARENT RC-8

**Title:** `RC-8 · Four bugs found by auditing what the software threw away`

**Labels:** `part-1` `bug` `done`
**Status:** Code complete, verification run in progress
**Priority:** High

**Description:**

**How these were found.** Rather than only checking what the software reported,
we looked at what it *discarded* and why. All four bugs below were silent — the
software produced sensible-looking output while being wrong. Three of them made
our own results look worse than they are; one made them look better.

---

**Bug 1 — Stones touching a rib were thrown away with the rib.**

Bone and stone are both bright. When a stone lies against a rib or the spine, the
blurry boundary between them can bridge the gap, so the software sees one
connected object rather than two. That object is mostly bone, so the "is this
bone?" rule discards it — **and the stone goes in the bin with it.** Nothing in
the output records that a stone was ever there.

We found this by noticing a discarded object that was only 84% bone. The other
16% had to be something else, and the report for that scan describes kidney
stones.

**Fix:** instead of discarding the whole object, split it. The part touching bone
is kept as its own candidate and still discarded (so the audit trail survives),
and the remaining part is re-examined on its own merits. Nothing is deleted, so
objects that genuinely are all bone behave exactly as before — only mixed cases
change.

---

**Bug 2 — We were penalised for correctly finding nothing.**

One report reads: *"The previously described 5 mm calculus in the interpolar
calyx of the right kidney **is not visualized on the present study**."* The stone
has passed. It is gone. Our software correctly found nothing — and was scored as
having **missed a stone**, because the scoring only looked for the words
"calculus … calyx … kidney" and ignored the "is not visualized".

The spreadsheet's own location column has the same blind spot: it labels that
scan as containing a renal stone. Both were presumably derived from the same
sentence.

**Fix:** recognise phrases meaning absence — "not visualized", "no longer seen",
"has passed", "resolution of", "no residual calculus" — and apply this to both
the spreadsheet column and our text search.

---

**Bug 3 — We compared our kidney measurement against a gallbladder stone.**

Our size accuracy was computed against "the largest stone size mentioned anywhere
in the report". In one scan the largest number in the report is a **13 mm
gallbladder stone** — a different organ entirely. In several others it is a
ureteric stone, outside our search area.

The symptom was a suspicious statistic: average error of 6.3 mm but a *typical*
error of only −1.5 mm. That gap was contamination from unrelated organs, not
measurement error.

**Fix:** only compare against sizes the report explicitly attaches to a stone
inside the kidney.

---

**Bug 4 — "Largest stone" included stones we had rejected.**

The per-scan summary computed "largest stone" and "total volume" over **all**
candidates including the rejected ones, while the stone *count* correctly used
only accepted ones. Results: one scan reported "0 stones" and "largest 16 mm" on
the same row. Another reported a 20.19 mm largest stone when its only real stone
is 2.69 mm — and the radiologist's report says 2 mm.

That last one matters for how we present ourselves: corrected, that scan shows
our software measuring a 2 mm stone to within 0.7 mm, which is good. Uncorrected,
it looked like a 18 mm error.

**Fix:** summarise accepted stones only.

---

**Also added:** the output now records the exact brightness value each
accept/reject decision was based on. Previously a rejection reading "not bright
enough" could sit next to a recorded peak of 575 HU, which looks like a
contradiction. It wasn't — the two numbers describe different regions — but
nobody could tell that without reading the source code. This also makes `RC-10`
possible without re-running anything.

**Remaining work on this issue**

- [ ] verification run completes (37 scans, ~4 min each)
- [ ] recompute all metrics and update the methods document
- [ ] confirm whether Bug 1's fix actually rescues stones in practice. It
      triggered correctly on the scan where we found it, but the rescued fragment
      turned out not to be bright enough to qualify as a stone — so it rescued
      nothing *there*. Whether it helps elsewhere is what this run will show.

**Code:** `utils/detect_stones.py`, `utils/compare_reports.py`
