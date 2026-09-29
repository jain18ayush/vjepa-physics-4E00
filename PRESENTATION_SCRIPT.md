# Speaker script

About fifteen minutes, then discussion. Fourteen slides. If you are long,
shorten slide 8 to two sentences and slide 10 to the speed number only.
Do not rush slides 6 and 7. Those are the steering result, and they are
the ones that need the plain-language version.

Open `presentation/index.html`. Space or the arrow keys advance. `N`
shows the cue for the current slide. `F` is fullscreen.

Say “readout” out loud. The papers say “probe.” They are the same thing:
a straight line from the model’s vector to the label. Say “copies” for
what the old slides called an axis or a subspace.

---

## Slide 1 — Title (0:00–0:30)

This is about three facts of a moving disk, direction, speed, and
acceleration, inside a frozen video model.

I do not train V-JEPA. I ask where those facts become readable, whether
each one is stored once or many times, and whether editing that storage
changes what a readout sees.

The ending is simple. Speed and acceleration are flat. Direction is a
circle, with some thickness around it.

---

## Slide 2 — The experiment (0:30–1:20)

Every video is one blue disk, sixteen frames, on a dark background.

There are three datasets. Fifteen hundred clips labeled by angle, half
coasting and half speeding up. Fifteen hundred labeled by speed, from a
quarter of a meter per second to four. Fifteen hundred labeled by
acceleration, starting from rest.

The questions are on the right. When can a readout see each variable? Is
it stored once, or copied? If I edit the copies, does the readout change?

---

## Slide 3 — The readout (1:20–2:20)

One piece of setup, used everywhere after this.

The model is V-JEPA 2, large, twenty-four layers. I leave it frozen and
I never run the predictor. Each clip goes in as the real sixteen frames.
Each layer returns a cloud of tokens. I average that cloud into one
vector of 1,024 numbers.

A probe, which I’ll call a readout, is a straight line from that vector
to the label. For direction I do not predict the degree number directly,
because 0° and 360° are the same direction. I predict sine and cosine,
then convert back to an angle. If you guess an angle at random, you are
off by 90° on average. That is the chance line.

Training fits the readout. Validation is how I pick a layer. Test clips
are not used for either. When I say “held out,” that is what I mean.

---

## Slide 4 — Layer 5 (2:20–3:30)

What I did: train a new readout at every layer, and pick the layer using
validation only.

What I found, and you can see it in the blue curves: speed and
acceleration are already readable at layer 0. R² is about 0.80. R² near
1 means the readout has the label. Direction, on the clips that coast at
constant velocity, is weak at the start, R² 0.23, and jumps to 0.93 at
layer 5. It stays high after that.

Everything from here on is at layer 5. The paper saw the same jump, at
layer 8 of this same model. I do not see that drop at the last layer.
Their videos were more complicated than one disk.

---

## Slide 5 — How many copies (3:30–4:40)

What I did, still at layer 5: train a direction readout, delete the part
of the vector it was using, train another one on what is left, and keep
going until a new readout can no longer see the angle.

Direction survives 41 rounds. Each readout has two outputs, sine and
cosine, so that is 82 numbers. The score wobbles as I go, which is the
sine and the cosine taking turns.

Speed is gone after 5 rounds. Acceleration after 2. Those declines are
smooth.

So direction is written down many times. Speed and acceleration are
written down a few times. I would not defend 82 as a precise constant.
I would defend the gap.

---

## Slide 6 — What the edit is (4:40–6:20)

This is the experiment that used to be phrased as “one axis moves the
angle.” Here is the same thing in the operations I actually ran.

I take test clips the readouts have never seen. I want the readout to
say 90°, no matter which way the disk was really going. I am allowed to
change the layer-5 vector only along the direction probes from the last
slide. Everything else in the vector stays put. Then a new readout,
trained only on unedited test clips, tells us what angle it sees.

Read the three cards left to right.

No edit. The new readout still knows the video. It is 19° off the real
angle. It is 90° away from 90°. That second number is not a failure. The
real angles are scattered all around the circle, so a fixed target of 90°
is, on average, a quarter turn away.

Edit along one probe. I change the vector in the one direction that
single readout uses. The reported angle moves toward 90°, now 58° away,
and it starts to forget the real angle, now 38° off. It moved. It did
not arrive. One copy of the angle is not the whole store.

Edit five probes together. Now the new readout is 11° from 90° and 82°
from the angle that was actually in the video. The copies agree on the
new angle, so the fresh readout follows them.

“Axis” meant that one direction in the vector. “Subspace” meant several
of them, edited together.

---

## Slide 7 — How to read the table (6:20–7:20)

Same experiment, more copies. The left column is “how far is the readout
from the angle I asked for.” The right column is “how far is it from the
angle in the video.”

You want the left number to fall and the right number to rise. That pair
of movements means I placed the representation on a specific angle. If
both numbers got worse, I would have just damaged the vector.

By 20 copies the readout is 6° from 90° and 89° from the original angle.
All 41 copies get the target error to 4°.

A concrete check: if a disk was really moving at 10°, and I perfectly
rewrote it as 90°, the left column would be about 0° and the right column
about 80°. I am in that regime from five probes onward.

Coasting clips and speeding-up clips do the same thing. The paper needed
about 20 probes to get within 12° of the target. I am there by five.

---

## Slide 8 — The paper (7:20–7:50)

Four agreements. Speed and acceleration show up immediately. Direction
shows up in a jump, earlier here than in their figure. Direction takes
many probes to erase, speed takes a few. Editing many probes on held-out
clips moves the readout onto a chosen angle.

The videos are simpler, so the numbers are easier. The pattern is the
same.

The open question is the rest of the talk. Those many probes could be
many different facts, or many copies of one circle.

---

## Slide 9 — Curve versus straight line (7:50–9:10)

What I did: at layer 5, average the training vectors that share an
angle, and connect the averages with a smooth loop. Fit that loop only
on coasting clips. Also draw the straight line between the 0° average
and the 180° average. Put each test clip onto each path, and read the
angle with a readout that never saw an edited vector.

Point at the plot. The teal line follows the ramp from 0 to 180. Halfway,
it is 5.5° off the angle it is supposed to be passing through. The red
line is the straight line. It cuts through the inside of the circle, and
the readout goes the wrong way around. Halfway, it is 171° off.

They meet at the end, because the end is the same point, and they are
both 18.6° off 180° under this readout. Editing 20 copies toward 180°,
with no curve at all, lands at 8.5°. Tighter landing. It does not visit
the angles in between.

So the circle is how the model passes through directions. The pile of
copies is how you force one particular direction.

---

## Slide 10 — Speed (9:10–9:50)

Same two paths for speed, from half a meter per second to 3.5.

Halfway, the straight line is off by a tenth of a meter per second. The
curve is off by about four tenths. The curve is not buying anything.
Both end at 2.9 rather than 3.5, so this readout and the average vector
for 3.5 m/s do not quite agree. That is a limit of the readout, and it
is the same for both paths.

Acceleration is the same shape, and I would not lean on its degree-style
errors: that readout’s R² is 0.33.

One direction control belongs here. A short hop, 22° around to 338°,
never needs to cut through the circle. Curve and straight line both
finish 1.3° off. Twenty copies finish at 8.4°, worse, because that edit
is a big rewrite for a small move.

---

## Slide 11 — Bending the path (9:50–10:50)

If the curve were the only set of vectors that represent those angles,
then an optimizer chasing the readout should fall back onto the curve.

What I did: start on the straight line, and nudge the path until the
readout follows 0° to 180° at every step.

It succeeds. Error is about 1°, middle and end.

It does not return to the curve. It ends up farther away than the
straight line was, distance 3.3 versus 1.6. Walking the curve itself, in
this run, is about 12° off. More orderly, worse score.

So the curve is a good route through the angles, and it is not the only
set of vectors the readout will accept. There is room beside the curve.
That room is the next slide.

If a writeup on the table says the curve’s final error here is 8.8°,
use the result file: the unoptimized endpoint is 19.8°, same as the
straight line, because they share that endpoint.

---

## Slide 12 — The sideways nudge (10:50–12:10)

What I did: take each training vector, pull it onto the nearest point
of the curve, and delete the part that points along the curve. Whatever
is left sticks out sideways. To steer, I walk along the curve so the
angle passes through the intermediates, and I add a small sideways
nudge chosen so the readout hits the final angle.

What I found: almost all the scatter is sideways. Energy 26 sticking
out, against 0.03 along the curve.

On the left plot, the final error falls off a cliff at two sideways
directions. The curve alone ends about 20° off. Curve plus two sideways
directions ends at 2.8°. Editing 20 copies with no curve, measured in
this same run, ends at 3.5°.

The right plot is the check that I did not recreate the straight-line
jump. The nudged path still follows the ramp. The red line still does
not.

So the practical edit is: follow the circle, then spend about two
sideways directions to land. You do not need all 82 copies once the
circle is pinned down.

---

## Slide 13 — What the sideways part contains (12:10–13:20)

What I did: predict the labels from position along the curve, and
separately from the sideways coordinates.

The curve knows the angle, R² 0.55, and does not know the speed. The
sideways part knows the speed, R² 0.70, and does not know the angle.
That angle R² is negative, which means it is worse than guessing the
average.

A step along the curve moves the decoded angle by about 9° per unit. A
step on the second sideways direction moves it by a tenth of a degree.
So that direction really is a level set of angle: you can move along it
and the angle stays put.

Removing speed from the vectors before refitting only shrinks the
sideways energy by 28%. Most of the thickness remains. The sideways
part is partly speed, and partly extra copies of the same angle.

---

## Slide 14 — Depth, and the close (13:20–15:00)

Last measurement. I rebuild that curve at layers 0, 5, 12, and 23.

The sideways energy grows from 3 to 26 to 50 to 221. Over the same
stretch, a step along the curve moves the angle by 29° at the first
layer and by less than a degree at the last. After direction becomes
readable, the model does not collapse it onto a wire. It keeps more
ways to write the same angle.

The recipe I would actually use: walk the curve when the angle should
pass through intermediates. Add about two sideways directions when you
need to hit one angle exactly. Do not fit direction and speed as two
unrelated curves. They share the sideways part.

What this does not show: edited pixels, or the video predictor agreeing
with the readout. The thing I edited is the averaged vector, and the
judge is a linear readout.

---

## If you are asked

**What was an “axis”?** One probe’s direction inside the 1,024-number
vector. Editing one of them changes one copy of the angle. The other
copies still vote for the original, so a fresh readout only moves part
of the way.

**Why is “no edit” 90° from the target and 19° from the truth?** Because
the readout is working. It reports the real angle, and the real angles
are spread around the circle, so they sit 90° from a fixed target of 90°
on average.

**Why does the curve lose at 180° and win on a short hop?** At the end of
a half turn, the average vector and the global readout disagree by about
19°. The multi-copy edit keeps pushing until that readout complies. On a
short hop the average is already a good target, and the big edit
overshoots it.

**Why did the optimizer leave the curve?** The readout only cares that
the angle is right. Many vectors off the curve still have the right
angle, especially the sideways directions that carry speed and spare
copies.
