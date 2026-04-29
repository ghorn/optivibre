# Multikite Model Parity Ledger

This file records the source-backed parity surface for the Rust model. The goal
is to keep control-system debugging separate from physics uncertainty.

## Source Pins

- Physics equations: `reference_source@2052ae8e69af45be8a2aee4eee14edd9c88ff68f`, the parent of the multikite removal commit.
- Controller and fit-based Reference assets: `reference_source@e18990d549770ad114dceba62548c9bb0dce8be8`.
- The asset manifest stores the pre-removal physics commit, not the removal commit.

## Parity Gates

The unit tests in `src/model.rs` pin these source equations with hard-coded
golden values:

- `sim/models/src/Kitty/Models/Tether.hs:tetherModel`
  - Half-length end segments.
  - Spring and damping force signs.
  - Midpoint tangent drag.
  - Endpoint forces and top tension.
  - Tether strain energy and dissipated power.
- `sim/models/src/Kitty/Models/Bridle.hs:bridleModel`
  - X/Z-plane bridle-ring projection about the pivot.
  - Tangential bridle velocity projection.
- `kittybutt/core/src/Kitty/Models/Aero/AeroCoeffs.hs:evalAeroCoeffs`
  - Reference AVL nominal quartic fit.
  - P/Q/R rate scaling by `[b, c, b] / (2 V)`.
  - Direct summation of control-surface polynomial terms.
- `kittybutt/core/src/Kitty/Models/Aero/AeroFrames.hs` plus
  `sim/models/src/Kitty/Models/Aircraft.hs`
  - CAD-point apparent wind, alpha, and beta.
  - Wind-axis basis matching `unsafeToDcmC2W`.
  - `qbar * Sref` force scaling.
  - `[b, c, b]` aerodynamic moment scaling.
  - Body-origin moment shift from the CAD offset.
- `kittybutt/core/src/Kitty/Models/Rotors.hs:rotorModel`
  - Reference XROTOR thrust and torque quartic fits.
  - Rotor acceleration from `motorTorque - aeroTorque`.
  - Motor power from `motorTorque * motorSpeed`.

Run:

```sh
cargo test -p multikite_sim model::tests
```

## Known Intentional Splits

- Rotor parity currently follows the 2016 fit-based `kittybutt/core`
  `rotorModel`. The older pre-removal `sim/models` rotor path had table-driven
  thrust, normal force, torque, and wake terms. Those older table terms are not
  yet vendored into `multikite_sim`.
- Motor parity is not exact against the pre-removal full aircraft stack. The
  Haskell `Aircraft.hs` path routes rotor aero torque through `motorModel`,
  battery bus voltage/current limits, motor signs, and gyroscopic motor moments.
  The Rust milestone model currently takes motor torque as the actuator command
  and applies the fit-based rotor acceleration directly.
- Non-rotor aero parity currently follows the 2016 fit-based AVL model assets
  through body-frame force/moment generation. The older pre-removal `sim/models`
  path could choose lookup tables, strip model, or crossfade depending on model
  options. Those branches are intentionally out of scope; the Rust runtime should
  terminate outside the fit's valid operating region instead of switching to the
  strip/crossfade model.
- Rigid-body translation/rotation and CAD-to-body force relocation were audited
  against `RigidBody.hs`, `Aircraft.hs`, and `Kite.hs`. The implemented equations
  match the diagonal-inertia path used here, including CAD velocity for
  aerodynamics and bridle/tether moments about the body origin. The rotor moment
  relocation currently has no numerical effect for the Reference fixed `+X` rotor
  force, but should be revisited if tilted/off-axis rotors are added.
- The parity gates cover the plant model. Controller correctness is intentionally
  not inferred from these tests.

## Current Diagnostic Note

The pinned Reference XROTOR fit returns negative thrust at `V = 23.5 m/s` and
`omega = 335 rad/s` with the current `+X` rotor axis. That is now an explicit
test fixture, so any thrust-direction change should be made deliberately against
the source model rather than by tuning around a visual symptom.
