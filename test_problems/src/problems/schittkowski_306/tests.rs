use super::helpers::*;

#[test]
fn source_backed_cases_are_complete_below_200() {
    let mut ids: Vec<_> = super::cases()
        .into_iter()
        .map(|case| case.id)
        .filter_map(|id| id.strip_prefix("schittkowski_tp"))
        .map(|suffix| suffix.parse::<u16>().expect("numeric Schittkowski suffix"))
        .filter(|id| *id < 200)
        .collect();
    ids.sort_unstable();

    let mut expected: Vec<u16> = (1..=119).collect();
    expected.retain(|id| !matches!(id, 82 | 94 | 115));

    assert_eq!(ids, expected);
    assert!(ids.contains(&119));
    assert!(!ids.iter().any(|id| (120..=200).contains(id)));
}

#[test]
fn source_objective_values_match_validated_values() {
    assert!((rosenbrock_value(1.0, 1.0) - 0.0).abs() <= 1e-14);
    let [x2, y2] = tp002_solution();
    assert!((tp002_objective() - rosenbrock_value(x2, y2)).abs() <= 1e-14);
    assert!((tp003_value(0.0, 0.0) - 0.0).abs() <= 1e-14);
    assert!((tp004_value(1.0, 0.0) - 8.0 / 3.0).abs() <= 1e-14);
    let [x5, y5] = tp005_solution();
    let f5 = tp005_value(x5, y5);
    assert!((f5 - (-3.0_f64.sqrt() / 2.0 - std::f64::consts::PI / 3.0)).abs() <= 1e-12);
    assert!((tp006_value(1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!((tp006_eq(1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!((tp007_value(0.0, 3.0_f64.sqrt()) + 3.0_f64.sqrt()).abs() <= 1e-14);
    assert!((tp007_eq(0.0, 3.0_f64.sqrt()) - 0.0).abs() <= 1e-14);
    let [x8, y8] = tp008_solution();
    assert!((tp008_eq1(x8, y8) - 0.0).abs() <= 1e-14);
    assert!((tp008_eq2(x8, y8) - 0.0).abs() <= 1e-14);
    assert!((tp009_value(-3.0, -4.0) + 0.5).abs() <= 1e-14);
    assert!((tp009_eq(-3.0, -4.0) - 0.0).abs() <= 1e-14);
    assert!((tp010_value(0.0, 1.0) + 1.0).abs() <= 1e-14);
    assert!((tp010_ineq(0.0, 1.0) - 0.0).abs() <= 1e-14);
    let [x11, y11] = tp011_solution();
    assert!((tp011_value(x11, y11) - tp011_objective()).abs() <= 1e-12);
    assert!(tp011_ineq(x11, y11).abs() <= 1e-10);
    assert!((tp012_value(2.0, 3.0) + 30.0).abs() <= 1e-14);
    assert!((tp012_ineq(2.0, 3.0) - 0.0).abs() <= 1e-14);
    assert!((tp013_value(1.0, 0.0) - 1.0).abs() <= 1e-14);
    assert!((tp013_ineq(1.0, 0.0) - 0.0).abs() <= 1e-14);
    let [x14, y14] = tp014_solution();
    assert!((tp014_value(x14, y14) - (9.0 - 23.0 * 7.0_f64.sqrt() / 8.0)).abs() <= 1e-14);
    assert!(tp014_eq(x14, y14).abs() <= 1e-14);
    assert!(tp014_ineq(x14, y14).abs() <= 1e-14);
    assert!((tp015_value(0.5, 2.0) - 3.065).abs() <= 1e-14);
    assert!((tp015_ineq1(0.5, 2.0) - 0.0).abs() <= 1e-14);
    assert!((tp016_value(0.5, 0.25) - 0.25).abs() <= 1e-14);
    assert!((tp017_value(0.0, 0.0) - 1.0).abs() <= 1e-14);
    let [x18, y18] = tp018_solution();
    assert!((tp018_value(x18, y18) - 5.0).abs() <= 1e-14);
    assert!(tp018_ineq1(x18, y18).abs() <= 1e-12);
    let [x19, y19] = tp019_solution();
    assert!((tp019_value(x19, y19) - tp019_objective()).abs() <= 1e-9);
    assert!(tp019_ineq2(x19, y19).abs() <= 1e-9);
    assert!(
        (tp020_value(0.5, 3.0_f64.sqrt() * 0.5) - (81.5 - 25.0 * 3.0_f64.sqrt())).abs() <= 1e-12
    );
    assert!((tp021_value(2.0, 0.0) + 99.96).abs() <= 1e-14);
    assert!((tp021_ineq(2.0, 0.0) + 10.0).abs() <= 1e-14);
    assert!((tp022_value(1.0, 1.0) - 1.0).abs() <= 1e-14);
    assert!((tp022_ineq1(1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!((tp023_value(1.0, 1.0) - 2.0).abs() <= 1e-14);
    assert!((tp023_ineq1(1.0, 1.0) + 1.0).abs() <= 1e-14);
    assert!((tp024_value(3.0, 3.0_f64.sqrt()) + 1.0).abs() <= 1e-14);
    assert!(tp024_ineq1(3.0, 3.0_f64.sqrt()).abs() <= 1e-14);
    assert!(tp025_value(50.0, 25.0, 1.5) <= 1e-24);
    assert!((tp026_value(1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!((tp026_eq(1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!((tp027_value(-1.0, 1.0, 0.0) - 4.0).abs() <= 1e-14);
    assert!((tp027_eq(-1.0, 1.0, 0.0) - 0.0).abs() <= 1e-14);
    assert!((tp028_value(0.5, -0.5, 0.5) - 0.0).abs() <= 1e-14);
    assert!((tp028_eq(0.5, -0.5, 0.5) - 0.0).abs() <= 1e-14);
    assert!((tp029_value(4.0, 2.0 * 2.0_f64.sqrt(), 2.0) + 16.0 * 2.0_f64.sqrt()).abs() <= 1e-14);
    assert!((tp029_ineq(4.0, 2.0 * 2.0_f64.sqrt(), 2.0) - 0.0).abs() <= 1e-14);
    assert!((tp030_value(1.0, 0.0, 0.0) - 1.0).abs() <= 1e-14);
    assert!((tp030_ineq(1.0, 0.0) - 0.0).abs() <= 1e-14);
    assert!((tp031_value(1.0 / 3.0_f64.sqrt(), 3.0_f64.sqrt(), 0.0) - 6.0).abs() <= 1e-14);
    assert!((tp031_ineq(1.0 / 3.0_f64.sqrt(), 3.0_f64.sqrt()) - 0.0).abs() <= 1e-14);
    assert!((tp032_value(0.0, 0.0, 1.0) - 1.0).abs() <= 1e-14);
    assert!((tp032_eq(0.0, 0.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!((tp032_ineq(0.0, 0.0, 1.0) + 1.0).abs() <= 1e-14);
    assert!(
        (tp033_value(0.0, 2.0_f64.sqrt(), 2.0_f64.sqrt()) - (2.0_f64.sqrt() - 6.0)).abs() <= 1e-14
    );
    assert!(tp033_ineq1(0.0, 2.0_f64.sqrt(), 2.0_f64.sqrt()).abs() <= 1e-14);
    assert!((tp034_value(10.0_f64.ln().ln()) + 10.0_f64.ln().ln()).abs() <= 1e-14);
    assert!((tp034_ineq1(10.0_f64.ln().ln(), 10.0_f64.ln()) - 0.0).abs() <= 1e-14);
    assert!((tp035_value(4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0) - 1.0 / 9.0).abs() <= 1e-14);
    assert!((tp035_ineq(4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0) - 0.0).abs() <= 1e-14);
    assert!((tp036_value(20.0, 11.0, 15.0) + 3300.0).abs() <= 1e-14);
    assert!((tp036_ineq(20.0, 11.0, 15.0) - 0.0).abs() <= 1e-14);
    assert!((tp037_value(24.0, 12.0, 12.0) + 3456.0).abs() <= 1e-14);
    assert!((tp037_ineq1(24.0, 12.0, 12.0) - 0.0).abs() <= 1e-14);
    assert!((tp038_value(1.0, 1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!((tp039_value(1.0) + 1.0).abs() <= 1e-14);
    assert!((tp039_eq1(1.0, 1.0, 0.0) - 0.0).abs() <= 1e-14);
    assert!((tp039_eq2(1.0, 1.0, 0.0) - 0.0).abs() <= 1e-14);
    assert!(
        (tp040_value(
            2.0_f64.powf(-1.0 / 3.0),
            2.0_f64.powf(-0.5),
            2.0_f64.powf(-11.0 / 12.0),
            2.0_f64.powf(-0.25)
        ) + 0.25)
            .abs()
            <= 1e-14
    );
    assert!((tp041_value(2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) - 52.0 / 27.0).abs() <= 1e-14);
    assert!((tp041_eq(2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0) - 0.0).abs() <= 1e-14);
    assert!(
        (tp042_value(2.0, 2.0, 0.72_f64.sqrt(), 1.28_f64.sqrt()) - (28.0 - 10.0 * 2.0_f64.sqrt()))
            .abs()
            <= 1e-14
    );
    assert!((tp042_eq1(2.0) - 0.0).abs() <= 1e-14);
    assert!((tp042_eq2(0.72_f64.sqrt(), 1.28_f64.sqrt()) - 0.0).abs() <= 1e-14);
    assert!((tp043_value(0.0, 1.0, 2.0, -1.0) + 44.0).abs() <= 1e-14);
    assert!((tp043_ineq1(0.0, 1.0, 2.0, -1.0) - 0.0).abs() <= 1e-14);
    assert!((tp044_value(0.0, 3.0, 0.0, 4.0) + 15.0).abs() <= 1e-14);
    assert!((tp044_ineq3(0.0, 3.0) - 0.0).abs() <= 1e-14);
    assert!((tp045_value(1.0, 2.0, 3.0, 4.0, 5.0) - 1.0).abs() <= 1e-14);
    assert!((tp046_value(1.0, 1.0, 1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!(tp046_eqs([1.0; 5]).into_iter().all(|v| v.abs() <= 1e-14));
    assert!((tp047_value(1.0, 1.0, 1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!(tp047_eqs([1.0; 5]).into_iter().all(|v| v.abs() <= 1e-14));
    assert!((tp048_value(1.0, 1.0, 1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!(tp048_eqs([1.0; 5]).into_iter().all(|v| v.abs() <= 1e-14));
    assert!((tp049_value(1.0, 1.0, 1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!(tp049_eqs([1.0; 5]).into_iter().all(|v| v.abs() <= 1e-14));
    assert!((tp050_value(1.0, 1.0, 1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!(tp050_eqs([1.0; 5]).into_iter().all(|v| v.abs() <= 1e-14));
    assert!((tp051_value(1.0, 1.0, 1.0, 1.0, 1.0) - 0.0).abs() <= 1e-14);
    assert!(tp051_eqs([1.0; 5]).into_iter().all(|v| v.abs() <= 1e-14));
    let tp52 = [
        -33.0 / 349.0,
        11.0 / 349.0,
        180.0 / 349.0,
        -158.0 / 349.0,
        11.0 / 349.0,
    ];
    assert!((tp052_value(tp52) - 1859.0 / 349.0).abs() <= 1e-12);
    assert!(tp052_eqs(tp52).into_iter().all(|v| v.abs() <= 1e-14));
    let tp53 = [
        -33.0 / 43.0,
        11.0 / 43.0,
        27.0 / 43.0,
        -5.0 / 43.0,
        11.0 / 43.0,
    ];
    assert!((tp053_value(tp53) - 176.0 / 43.0).abs() <= 1e-12);
    assert!(tp052_eqs(tp53).into_iter().all(|v| v.abs() <= 1e-14));
    let tp54 = [9.16e4 / 7.0, 79.0 / 70.0, 2.0e6, 10.0, 1.0e-3, 1.0e8];
    assert!((tp054_value(tp54) + (-27.0 / 280.0_f64).exp()).abs() <= 1e-12);
    assert!((tp054_eq(tp54) - 0.0).abs() <= 1e-9);
    let tp55 = [0.0, 4.0 / 3.0, 5.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0];
    assert!((tp055_value(tp55) - 19.0 / 3.0).abs() <= 1e-14);
    assert!(tp055_eqs(tp55).into_iter().all(|v| v.abs() <= 1e-14));
    let tp56 = [
        2.4,
        1.2,
        1.2,
        (4.0_f64 / 7.0).sqrt().asin(),
        (2.0_f64 / 7.0).sqrt().asin(),
        (2.0_f64 / 7.0).sqrt().asin(),
        std::f64::consts::FRAC_PI_2,
    ];
    assert!((tp056_value(tp56) + 3.456).abs() <= 1e-14);
    assert!(tp056_eqs(tp56).into_iter().all(|v| v.abs() <= 1e-14));
    let tp57 = [0.419952674511, 1.28484562930];
    assert!((tp057_value(tp57[0], tp57[1]) - 0.0284596697213).abs() <= 1e-11);
    assert!(tp057_ineq(tp57[0], tp57[1]) <= 1e-10);
    let tp58 = [-0.786150483331, 0.618034533851];
    assert!((rosenbrock_value(tp58[0], tp58[1]) - 3.19033354957).abs() <= 1e-10);
    assert!(tp058_ineqs(tp58[0], tp58[1]).into_iter().all(|v| v <= 1e-5));
    let tp59 = [13.5501042366, 51.6601812877];
    assert!((tp059_value(tp59[0], tp59[1]) + 7.80422632408).abs() <= 1e-7);
    assert!(tp059_ineqs(tp59[0], tp59[1]).into_iter().all(|v| v <= 1e-7));
    let tp60 = [1.10485902423, 1.19667419413, 1.53526225739];
    assert!((tp060_value(tp60) - 0.0325682002513).abs() <= 1e-11);
    assert!(tp060_eq(tp60).abs() <= 1e-9);
    let tp61 = [5.32677015744, -2.11899863998, 3.21046423906];
    assert!((tp061_value(tp61) + 143.646142201).abs() <= 1e-8);
    assert!(tp061_eqs(tp61).into_iter().all(|v| v.abs() <= 1e-9));
    let tp62 = [0.617813298210, 0.328202155786, 0.0539845460119];
    assert!((tp062_value(tp62) + 26272.5144873).abs() <= 1e-6);
    assert!(tp062_eq(tp62).abs() <= 1e-10);
    let tp63 = [3.51211841492, 0.216988174172, 3.55217403459];
    assert!((tp063_value(tp63) - 961.715172127).abs() <= 1e-8);
    assert!(tp063_eqs(tp63).into_iter().all(|v| v.abs() <= 1e-9));
    let tp64 = [108.734717597, 85.1261394257, 204.324707858];
    assert!((tp064_value(tp64) - 6299.84242821).abs() <= 1e-6);
    assert!(tp064_ineq(tp64) <= 1e-10);
    let tp65 = [3.65046182158, 3.65046168940, 4.62041750754];
    assert!((tp065_value(tp65) - 0.953528856757).abs() <= 1e-10);
    assert!(tp065_ineq(tp65) <= 1e-8);
    let tp66 = [0.184126487951, 1.20216787321, 3.32732232258];
    assert!((tp066_value(tp66) - 0.518163274159).abs() <= 1e-10);
    assert!(tp066_ineqs(tp66).into_iter().all(|v| v <= 1e-9));
    let tp67 = [1728.37128614, 16000.0, 98.1415140238];
    assert!((tp067_value(tp67) + 1162.03650728).abs() <= 2e-2);
    assert!(tp067_ineqs(tp67).into_iter().all(|v| v <= 1e-2));
}

#[test]
fn documented_starting_points_evaluate_finitely() {
    let values = [
        rosenbrock_value(-2.0, 1.0),
        tp003_value(10.0, 1.0),
        tp004_value(1.125, 0.125),
        tp005_value(0.0, 0.0),
        tp006_value(-1.2, 1.0),
        tp006_eq(-1.2, 1.0),
        tp007_value(2.0, 2.0),
        tp007_eq(2.0, 2.0),
        tp008_eq1(2.0, 1.0),
        tp008_eq2(2.0, 1.0),
        tp009_value(0.0, 0.0),
        tp009_eq(0.0, 0.0),
        tp010_value(-10.0, 10.0),
        tp010_ineq(-10.0, 10.0),
        tp011_value(4.9, 0.1),
        tp011_ineq(4.9, 0.1),
        tp012_value(0.0, 0.0),
        tp012_ineq(0.0, 0.0),
        tp013_value(0.0, 0.0),
        tp013_ineq(0.0, 0.0),
        tp014_value(2.0, 2.0),
        tp014_eq(2.0, 2.0),
        tp014_ineq(2.0, 2.0),
        tp015_value(-2.0, 1.0),
        tp015_ineq1(-2.0, 1.0),
        tp016_value(-2.0, 1.0),
        tp017_value(-2.0, 1.0),
        tp018_value(2.0, 2.0),
        tp019_value(20.1, 5.84),
        tp020_value(0.1, 1.0),
        tp021_value(2.0, -1.0),
        tp022_value(2.0, 2.0),
        tp023_value(3.0, 1.0),
        tp024_value(1.0, 0.5),
        tp025_value(100.0, 12.5, 3.0),
        tp026_value(-2.6, 2.0, 2.0),
        tp026_eq(-2.6, 2.0, 2.0),
        tp027_value(2.0, 2.0, 2.0),
        tp028_value(-4.0, 1.0, 1.0),
        tp029_value(1.0, 1.0, 1.0),
        tp030_value(1.0, 1.0, 1.0),
        tp031_value(1.0, 1.0, 1.0),
        tp032_value(0.1, 0.7, 0.2),
        tp033_value(0.0, 0.0, 3.0),
        tp034_value(0.0),
        tp035_value(0.5, 0.5, 0.5),
        tp036_value(10.0, 10.0, 10.0),
        tp037_value(10.0, 10.0, 10.0),
        tp038_value(-3.0, -1.0, -3.0, -1.0),
        tp039_value(2.0),
        tp039_eq1(2.0, 2.0, 2.0),
        tp039_eq2(2.0, 2.0, 2.0),
        tp040_value(0.8, 0.8, 0.8, 0.8),
        tp040_eq1(0.8, 0.8),
        tp040_eq2(0.8, 0.8, 0.8),
        tp040_eq3(0.8, 0.8),
        tp041_value(1.0, 1.0, 1.0),
        tp041_eq(1.0, 1.0, 1.0, 1.0),
        tp042_value(1.0, 1.0, 1.0, 1.0),
        tp042_eq1(1.0),
        tp042_eq2(1.0, 1.0),
        tp043_value(0.0, 0.0, 0.0, 0.0),
        tp044_value(0.0, 0.0, 0.0, 0.0),
        tp045_value(2.0, 2.0, 2.0, 2.0, 2.0),
        tp046_value(0.5 * 2.0_f64.sqrt(), 1.75, 0.5, 2.0, 2.0),
        tp047_value(2.0, 2.0_f64.sqrt(), -1.0, 2.0 - 2.0_f64.sqrt(), 0.5),
        tp048_value(3.0, 5.0, -3.0, 2.0, -2.0),
        tp049_value(10.0, 7.0, 2.0, -3.0, 0.8),
        tp050_value(35.0, -31.0, 11.0, 5.0, -5.0),
        tp051_value(2.5, 0.5, 2.0, -1.0, 0.5),
        tp052_value([2.0; 5]),
        tp053_value([2.0; 5]),
        tp054_value([6.0e3, 1.5, 4.0e6, 2.0, 3.0e-3, 5.0e7]),
        tp055_value([1.0, 2.0, 0.0, 0.0, 0.0, 2.0]),
        tp056_value([
            1.0,
            1.0,
            1.0,
            (1.0_f64 / 4.2).sqrt().asin(),
            (1.0_f64 / 4.2).sqrt().asin(),
            (1.0_f64 / 4.2).sqrt().asin(),
            (5.0_f64 / 7.2).sqrt().asin(),
        ]),
        tp057_value(0.42, 5.0),
        rosenbrock_value(-2.0, 1.0),
        tp059_value(90.0, 10.0),
        tp060_value([2.0; 3]),
        tp061_value([0.0; 3]),
        tp062_value([0.7, 0.2, 0.1]),
        tp063_value([2.0; 3]),
        tp064_value([1.0; 3]),
        tp065_value([-5.0, 5.0, 0.0]),
        tp066_value([0.0, 1.05, 2.9]),
        tp067_value([1.745e3, 1.2e4, 110.0]),
    ];
    assert!(values.into_iter().all(f64::is_finite));
}

fn rosenbrock_value(x: f64, y: f64) -> f64 {
    100.0 * (y - x.powi(2)).powi(2) + (1.0 - x).powi(2)
}

fn tp003_value(x: f64, y: f64) -> f64 {
    y + 1e-5 * (y - x).powi(2)
}

fn tp004_value(x: f64, y: f64) -> f64 {
    (x + 1.0).powi(3) / 3.0 + y
}

fn tp005_value(x: f64, y: f64) -> f64 {
    (x + y).sin() + (x - y).powi(2) - 1.5 * x + 2.5 * y + 1.0
}

fn tp006_value(x: f64, _y: f64) -> f64 {
    (1.0 - x).powi(2)
}

fn tp006_eq(x: f64, y: f64) -> f64 {
    10.0 * (y - x.powi(2))
}

fn tp007_value(x: f64, y: f64) -> f64 {
    (1.0 + x.powi(2)).ln() - y
}

fn tp007_eq(x: f64, y: f64) -> f64 {
    (1.0 + x.powi(2)).powi(2) + y.powi(2) - 4.0
}

fn tp008_solution() -> [f64; 2] {
    let x = ((25.0 + 301.0_f64.sqrt()) / 2.0).sqrt();
    [x, 9.0 / x]
}

fn tp008_eq1(x: f64, y: f64) -> f64 {
    x.powi(2) + y.powi(2) - 25.0
}

fn tp008_eq2(x: f64, y: f64) -> f64 {
    x * y - 9.0
}

fn tp009_value(x: f64, y: f64) -> f64 {
    (std::f64::consts::PI * x / 12.0).sin() * (std::f64::consts::PI * y / 16.0).cos()
}

fn tp009_eq(x: f64, y: f64) -> f64 {
    4.0 * x - 3.0 * y
}

fn tp010_value(x: f64, y: f64) -> f64 {
    x - y
}

fn tp010_ineq(x: f64, y: f64) -> f64 {
    3.0 * x.powi(2) - 2.0 * x * y + y.powi(2) - 1.0
}

fn tp011_value(x: f64, y: f64) -> f64 {
    (x - 5.0).powi(2) + y.powi(2) - 25.0
}

fn tp011_ineq(x: f64, y: f64) -> f64 {
    x.powi(2) - y
}

fn tp012_value(x: f64, y: f64) -> f64 {
    0.5 * x.powi(2) + y.powi(2) - x * y - 7.0 * x - 7.0 * y
}

fn tp012_ineq(x: f64, y: f64) -> f64 {
    4.0 * x.powi(2) + y.powi(2) - 25.0
}

fn tp013_value(x: f64, y: f64) -> f64 {
    (x - 2.0).powi(2) + y.powi(2)
}

fn tp013_ineq(x: f64, y: f64) -> f64 {
    y - (1.0 - x).powi(3)
}

fn tp014_value(x: f64, y: f64) -> f64 {
    (x - 2.0).powi(2) + (y - 1.0).powi(2)
}

fn tp014_eq(x: f64, y: f64) -> f64 {
    x - 2.0 * y + 1.0
}

fn tp014_ineq(x: f64, y: f64) -> f64 {
    0.25 * x.powi(2) + y.powi(2) - 1.0
}

fn tp015_value(x: f64, y: f64) -> f64 {
    (y - x.powi(2)).powi(2) + 0.01 * (1.0 - x).powi(2)
}

fn tp015_ineq1(x: f64, y: f64) -> f64 {
    1.0 - x * y
}

fn tp016_value(x: f64, y: f64) -> f64 {
    rosenbrock_value(x, y)
}

fn tp017_value(x: f64, y: f64) -> f64 {
    rosenbrock_value(x, y)
}

fn tp018_value(x: f64, y: f64) -> f64 {
    0.01 * x.powi(2) + y.powi(2)
}

fn tp018_ineq1(x: f64, y: f64) -> f64 {
    25.0 - x * y
}

fn tp019_value(x: f64, y: f64) -> f64 {
    (x - 10.0).powi(3) + (y - 20.0).powi(3)
}

fn tp019_ineq2(x: f64, y: f64) -> f64 {
    (x - 6.0).powi(2) + (y - 5.0).powi(2) - 82.81
}

fn tp020_value(x: f64, y: f64) -> f64 {
    rosenbrock_value(x, y)
}

fn tp021_value(x: f64, y: f64) -> f64 {
    0.01 * x.powi(2) + y.powi(2) - 100.0
}

fn tp021_ineq(x: f64, y: f64) -> f64 {
    -10.0 * x + y + 10.0
}

fn tp022_value(x: f64, y: f64) -> f64 {
    (x - 2.0).powi(2) + (y - 1.0).powi(2)
}

fn tp022_ineq1(x: f64, y: f64) -> f64 {
    x + y - 2.0
}

fn tp023_value(x: f64, y: f64) -> f64 {
    x.powi(2) + y.powi(2)
}

fn tp023_ineq1(x: f64, y: f64) -> f64 {
    1.0 - x - y
}

fn tp024_value(x: f64, y: f64) -> f64 {
    ((x - 3.0).powi(2) - 9.0) * y.powi(3) / (27.0 * 3.0_f64.sqrt())
}

fn tp024_ineq1(x: f64, y: f64) -> f64 {
    y - x / 3.0_f64.sqrt()
}

fn tp025_value(x0: f64, x1: f64, x2: f64) -> f64 {
    let mut objective = 0.0;
    for i in 1..=99 {
        let i_float = i as f64;
        let u = 25.0 + (-50.0 * (0.01 * i_float).ln()).powf(2.0 / 3.0);
        let residual = (-(u - x1).powf(x2) / x0).exp() - 0.01 * i_float;
        objective += residual.powi(2);
    }
    objective
}

fn tp026_value(x0: f64, x1: f64, x2: f64) -> f64 {
    (x0 - x1).powi(2) + (x1 - x2).powi(4)
}

fn tp026_eq(x0: f64, x1: f64, x2: f64) -> f64 {
    x0 * (1.0 + x1.powi(2)) + x2.powi(4) - 3.0
}

fn tp027_value(x0: f64, x1: f64, _x2: f64) -> f64 {
    rosenbrock_value(x0, x1)
}

fn tp027_eq(x0: f64, _x1: f64, x2: f64) -> f64 {
    x0 + x2.powi(2) + 1.0
}

fn tp028_value(x0: f64, x1: f64, x2: f64) -> f64 {
    (x0 + x1).powi(2) + (x1 + x2).powi(2)
}

fn tp028_eq(x0: f64, x1: f64, x2: f64) -> f64 {
    x0 + 2.0 * x1 + 3.0 * x2 - 1.0
}

fn tp029_value(x0: f64, x1: f64, x2: f64) -> f64 {
    -x0 * x1 * x2
}

fn tp029_ineq(x0: f64, x1: f64, x2: f64) -> f64 {
    x0.powi(2) + 2.0 * x1.powi(2) + 4.0 * x2.powi(2) - 48.0
}

fn tp030_value(x0: f64, x1: f64, x2: f64) -> f64 {
    x0.powi(2) + x1.powi(2) + x2.powi(2)
}

fn tp030_ineq(x0: f64, x1: f64) -> f64 {
    1.0 - x0.powi(2) - x1.powi(2)
}

fn tp031_value(x0: f64, x1: f64, x2: f64) -> f64 {
    9.0 * x0.powi(2) + x1.powi(2) + 9.0 * x2.powi(2)
}

fn tp031_ineq(x0: f64, x1: f64) -> f64 {
    1.0 - x0 * x1
}

fn tp032_value(x0: f64, x1: f64, x2: f64) -> f64 {
    (x0 + 3.0 * x1 + x2).powi(2) + 4.0 * (x0 - x1).powi(2)
}

fn tp032_eq(x0: f64, x1: f64, x2: f64) -> f64 {
    1.0 - x0 - x1 - x2
}

fn tp032_ineq(x0: f64, x1: f64, x2: f64) -> f64 {
    x0.powi(3) - 6.0 * x1 - 4.0 * x2 + 3.0
}

fn tp033_value(x0: f64, _x1: f64, x2: f64) -> f64 {
    (x0 - 1.0) * (x0 - 2.0) * (x0 - 3.0) + x2
}

fn tp033_ineq1(x0: f64, x1: f64, x2: f64) -> f64 {
    x0.powi(2) + x1.powi(2) - x2.powi(2)
}

fn tp034_value(x0: f64) -> f64 {
    -x0
}

fn tp034_ineq1(x0: f64, x1: f64) -> f64 {
    x0.exp() - x1
}

fn tp035_value(x0: f64, x1: f64, x2: f64) -> f64 {
    9.0 - 8.0 * x0 - 6.0 * x1 - 4.0 * x2
        + 2.0 * x0.powi(2)
        + 2.0 * x1.powi(2)
        + x2.powi(2)
        + 2.0 * x0 * x1
        + 2.0 * x0 * x2
}

fn tp035_ineq(x0: f64, x1: f64, x2: f64) -> f64 {
    x0 + x1 + 2.0 * x2 - 3.0
}

fn tp036_value(x0: f64, x1: f64, x2: f64) -> f64 {
    -x0 * x1 * x2
}

fn tp036_ineq(x0: f64, x1: f64, x2: f64) -> f64 {
    x0 + 2.0 * x1 + 2.0 * x2 - 72.0
}

fn tp037_value(x0: f64, x1: f64, x2: f64) -> f64 {
    -x0 * x1 * x2
}

fn tp037_ineq1(x0: f64, x1: f64, x2: f64) -> f64 {
    x0 + 2.0 * x1 + 2.0 * x2 - 72.0
}

fn tp038_value(x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    100.0 * (x1 - x0.powi(2)).powi(2)
        + (1.0 - x0).powi(2)
        + 90.0 * (x3 - x2.powi(2)).powi(2)
        + (1.0 - x2).powi(2)
        + 10.1 * ((x1 - 1.0).powi(2) + (x3 - 1.0).powi(2))
        + 19.8 * (x1 - 1.0) * (x3 - 1.0)
}

fn tp039_value(x0: f64) -> f64 {
    -x0
}

fn tp039_eq1(x0: f64, x1: f64, x2: f64) -> f64 {
    x1 - x0.powi(3) - x2.powi(2)
}

fn tp039_eq2(x0: f64, x1: f64, x3: f64) -> f64 {
    x0.powi(2) - x1 - x3.powi(2)
}

fn tp040_value(x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    -x0 * x1 * x2 * x3
}

fn tp040_eq1(x0: f64, x1: f64) -> f64 {
    x0.powi(3) + x1.powi(2) - 1.0
}

fn tp040_eq2(x0: f64, x2: f64, x3: f64) -> f64 {
    x0.powi(2) * x3 - x2
}

fn tp040_eq3(x1: f64, x3: f64) -> f64 {
    x3.powi(2) - x1
}

fn tp041_value(x0: f64, x1: f64, x2: f64) -> f64 {
    2.0 - x0 * x1 * x2
}

fn tp041_eq(x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    x0 + 2.0 * x1 + 2.0 * x2 - x3
}

fn tp042_value(x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    (x0 - 1.0).powi(2) + (x1 - 2.0).powi(2) + (x2 - 3.0).powi(2) + (x3 - 4.0).powi(2)
}

fn tp042_eq1(x0: f64) -> f64 {
    x0 - 2.0
}

fn tp042_eq2(x2: f64, x3: f64) -> f64 {
    x2.powi(2) + x3.powi(2) - 2.0
}

fn tp043_value(x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    x0.powi(2) + x1.powi(2) + 2.0 * x2.powi(2) + x3.powi(2) - 5.0 * x0 - 5.0 * x1 - 21.0 * x2
        + 7.0 * x3
}

fn tp043_ineq1(x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    x0.powi(2) + x1.powi(2) + x2.powi(2) + x3.powi(2) + x0 - x1 + x2 - x3 - 8.0
}

fn tp044_value(x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    x0 - x1 - x2 - x0 * x2 + x0 * x3 + x1 * x2 - x1 * x3
}

fn tp044_ineq3(x0: f64, x1: f64) -> f64 {
    3.0 * x0 + 4.0 * x1 - 12.0
}

fn tp045_value(x0: f64, x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    2.0 - x0 * x1 * x2 * x3 * x4 / 120.0
}

fn tp046_value(x0: f64, x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    (x0 - x1).powi(2) + (x2 - 1.0).powi(2) + (x3 - 1.0).powi(4) + (x4 - 1.0).powi(6)
}

fn tp046_eqs(x: [f64; 5]) -> [f64; 2] {
    [
        x[0].powi(2) * x[3] + (x[3] - x[4]).sin() - 1.0,
        x[1] + x[2].powi(4) * x[3].powi(2) - 2.0,
    ]
}

fn tp047_value(x0: f64, x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    (x0 - x1).powi(2) + (x1 - x2).powi(2) + (x2 - x3).powi(4) + (x3 - x4).powi(4)
}

fn tp047_eqs(x: [f64; 5]) -> [f64; 3] {
    [
        x[0] + x[1].powi(2) + x[2].powi(3) - 3.0,
        x[1] - x[2].powi(2) + x[3] - 1.0,
        x[0] * x[4] - 1.0,
    ]
}

fn tp048_value(x0: f64, x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    (x0 - 1.0).powi(2) + (x1 - x2).powi(2) + (x3 - x4).powi(2)
}

fn tp048_eqs(x: [f64; 5]) -> [f64; 2] {
    [
        x.iter().sum::<f64>() - 5.0,
        x[2] - 2.0 * (x[3] + x[4]) + 3.0,
    ]
}

fn tp049_value(x0: f64, x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    (x0 - x1).powi(2) + (x2 - 1.0).powi(2) + (x3 - 1.0).powi(4) + (x4 - 1.0).powi(6)
}

fn tp049_eqs(x: [f64; 5]) -> [f64; 2] {
    [
        x[0] + x[1] + x[2] + 4.0 * x[3] - 7.0,
        x[2] + 5.0 * x[4] - 6.0,
    ]
}

fn tp050_value(x0: f64, x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    (x0 - x1).powi(2) + (x1 - x2).powi(2) + (x2 - x3).powi(4) + (x3 - x4).powi(4)
}

fn tp050_eqs(x: [f64; 5]) -> [f64; 3] {
    [
        x[0] + 2.0 * x[1] + 3.0 * x[2] - 6.0,
        x[1] + 2.0 * x[2] + 3.0 * x[3] - 6.0,
        x[2] + 2.0 * x[3] + 3.0 * x[4] - 6.0,
    ]
}

fn tp051_value(x0: f64, x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    (x0 - x1).powi(2) + (x1 + x2 - 2.0).powi(2) + (x3 - 1.0).powi(2) + (x4 - 1.0).powi(2)
}

fn tp051_eqs(x: [f64; 5]) -> [f64; 3] {
    [
        x[0] + 3.0 * x[1] - 4.0,
        x[2] + x[3] - 2.0 * x[4],
        x[1] - x[4],
    ]
}

fn tp052_value(x: [f64; 5]) -> f64 {
    (4.0 * x[0] - x[1]).powi(2)
        + (x[1] + x[2] - 2.0).powi(2)
        + (x[3] - 1.0).powi(2)
        + (x[4] - 1.0).powi(2)
}

fn tp052_eqs(x: [f64; 5]) -> [f64; 3] {
    [x[0] + 3.0 * x[1], x[2] + x[3] - 2.0 * x[4], x[1] - x[4]]
}

fn tp053_value(x: [f64; 5]) -> f64 {
    (x[0] - x[1]).powi(2)
        + (x[1] + x[2] - 2.0).powi(2)
        + (x[3] - 1.0).powi(2)
        + (x[4] - 1.0).powi(2)
}

fn tp054_value(x: [f64; 6]) -> f64 {
    let v0 = x[0] - 1.0e4;
    let v1 = x[1] - 1.0;
    let v2 = x[2] - 2.0e6;
    let v3 = x[3] - 10.0;
    let v4 = x[4] - 1.0e-3;
    let v5 = x[5] - 1.0e8;
    let q = (1.5625e-8 * v0.powi(2) + 5.0e-5 * v0 * v1 + v1.powi(2)) / 0.96
        + v2.powi(2) / 4.9e13
        + 4.0e-4 * v3.powi(2)
        + 4.0e2 * v4.powi(2)
        + 4.0e-18 * v5.powi(2);
    -(-0.5 * q).exp()
}

fn tp054_eq(x: [f64; 6]) -> f64 {
    x[0] + 4.0e3 * x[1] - 1.76e4
}

fn tp055_value(x: [f64; 6]) -> f64 {
    x[0] + 2.0 * x[1] + 4.0 * x[4] + (x[0] * x[3]).exp()
}

fn tp055_eqs(x: [f64; 6]) -> [f64; 6] {
    [
        x[0] + 2.0 * x[1] + 5.0 * x[4] - 6.0,
        x[0] + x[1] + x[2] - 3.0,
        x[3] + x[4] + x[5] - 2.0,
        x[0] + x[3] - 1.0,
        x[1] + x[4] - 2.0,
        x[2] + x[5] - 2.0,
    ]
}

fn tp056_value(x: [f64; 7]) -> f64 {
    -x[0] * x[1] * x[2]
}

fn tp056_eqs(x: [f64; 7]) -> [f64; 4] {
    [
        x[0] - 4.2 * x[3].sin().powi(2),
        x[1] - 4.2 * x[4].sin().powi(2),
        x[2] - 4.2 * x[5].sin().powi(2),
        x[0] + 2.0 * x[1] + 2.0 * x[2] - 7.2 * x[6].sin().powi(2),
    ]
}

fn tp057_value(x0: f64, x1: f64) -> f64 {
    let (a, b) = tp057_data();
    let mut objective = 0.0;
    for i in 0..44 {
        let residual = b[i] - x0 - (0.49 - x0) * (-(x1 * (a[i] - 8.0))).exp();
        objective += residual.powi(2);
    }
    objective
}

fn tp057_ineq(x0: f64, x1: f64) -> f64 {
    x0 * x1 - 0.49 * x1 + 0.09
}

fn tp058_ineqs(x0: f64, x1: f64) -> [f64; 3] {
    [
        x0 - x1.powi(2),
        x1 - x0.powi(2),
        1.0 - x0.powi(2) - x1.powi(2),
    ]
}

fn tp059_value(x0: f64, x1: f64) -> f64 {
    let x0_2 = x0.powi(2);
    let x0_3 = x0.powi(3);
    let x0_4 = x0.powi(4);
    let x1_2 = x1.powi(2);
    let x1_3 = x1.powi(3);
    let x1_4 = x1.powi(4);
    -75.196 + 3.8112 * x0 - 0.12694 * x0_2 + 2.0567e-3 * x0_3 - 1.0345e-5 * x0_4 + 6.8306 * x1
        - 3.0234e-2 * x0 * x1
        + 1.28134e-3 * x0_2 * x1
        - 3.5256e-5 * x0_3 * x1
        + 2.266e-7 * x0_4 * x1
        - 0.25645 * x1_2
        + 3.4604e-3 * x1_3
        - 1.3514e-5 * x1_4
        + 28.106 / (x1 + 1.0)
        + 5.2375e-6 * x0_2 * x1_2
        + 6.3e-8 * x0_3 * x1_2
        - 7.0e-10 * x0_3 * x1_3
        - 3.4054e-4 * x0 * x1_2
        + 1.6638e-6 * x0 * x1_3
        + 2.8673 * (5.0e-4 * x0 * x1).exp()
}

fn tp059_ineqs(x0: f64, x1: f64) -> [f64; 3] {
    [
        700.0 - x0 * x1,
        0.008 * x0.powi(2) - x1,
        5.0 * (x0 - 55.0) - (x1 - 50.0).powi(2),
    ]
}

fn tp060_value(x: [f64; 3]) -> f64 {
    (x[0] - 1.0).powi(2) + (x[0] - x[1]).powi(2) + (x[1] - x[2]).powi(4)
}

fn tp060_eq(x: [f64; 3]) -> f64 {
    x[0] * (1.0 + x[1].powi(2)) + x[2].powi(4) - 4.0 - 3.0 * 2.0_f64.sqrt()
}

fn tp061_value(x: [f64; 3]) -> f64 {
    4.0 * x[0].powi(2) + 2.0 * x[1].powi(2) + 2.0 * x[2].powi(2) - 33.0 * x[0] + 16.0 * x[1]
        - 24.0 * x[2]
}

fn tp061_eqs(x: [f64; 3]) -> [f64; 2] {
    [
        3.0 * x[0] - 2.0 * x[1].powi(2) - 7.0,
        4.0 * x[0] - x[2].powi(2) - 11.0,
    ]
}

fn tp062_value(x: [f64; 3]) -> f64 {
    let b3 = x[2] + 0.03;
    let c3 = 0.13 * x[2] + 0.03;
    let b2 = b3 + x[1];
    let c2 = b3 + 0.07 * x[1];
    let b1 = b2 + x[0];
    let c1 = b2 + 0.09 * x[0];
    -32.174 * (255.0 * (b1 / c1).ln() + 280.0 * (b2 / c2).ln() + 290.0 * (b3 / c3).ln())
}

fn tp062_eq(x: [f64; 3]) -> f64 {
    x[0] + x[1] + x[2] - 1.0
}

fn tp063_value(x: [f64; 3]) -> f64 {
    1000.0 - x[0].powi(2) - 2.0 * x[1].powi(2) - x[2].powi(2) - x[0] * x[1] - x[0] * x[2]
}

fn tp063_eqs(x: [f64; 3]) -> [f64; 2] {
    [
        8.0 * x[0] + 14.0 * x[1] + 7.0 * x[2] - 56.0,
        x[0].powi(2) + x[1].powi(2) + x[2].powi(2) - 25.0,
    ]
}

fn tp064_value(x: [f64; 3]) -> f64 {
    5.0 * x[0] + 5.0e4 / x[0] + 20.0 * x[1] + 7.2e4 / x[1] + 10.0 * x[2] + 1.44e5 / x[2]
}

fn tp064_ineq(x: [f64; 3]) -> f64 {
    -1.0 + 4.0 / x[0] + 32.0 / x[1] + 120.0 / x[2]
}

fn tp065_value(x: [f64; 3]) -> f64 {
    (x[0] - x[1]).powi(2) + ((x[0] + x[1] - 10.0) / 3.0).powi(2) + (x[2] - 5.0).powi(2)
}

fn tp065_ineq(x: [f64; 3]) -> f64 {
    x[0].powi(2) + x[1].powi(2) + x[2].powi(2) - 48.0
}

fn tp066_value(x: [f64; 3]) -> f64 {
    0.2 * x[2] - 0.8 * x[0]
}

fn tp066_ineqs(x: [f64; 3]) -> [f64; 2] {
    [x[0].exp() - x[1], x[1].exp() - x[2]]
}

fn tp067_value(x: [f64; 3]) -> f64 {
    let y = tp067_state(x);
    -(0.063 * y[0] * y[3] - 5.04 * x[0] - 3.36 * y[1] - 0.035 * x[1] - 10.0 * x[2])
}

fn tp067_ineqs(x: [f64; 3]) -> [f64; 14] {
    let y = tp067_state(x);
    [
        -y[0],
        -y[1],
        85.0 - y[2],
        90.0 - y[3],
        3.0 - y[4],
        0.01 - y[5],
        145.0 - y[6],
        y[0] - 5.0e3,
        y[1] - 2.0e3,
        y[2] - 93.0,
        y[3] - 95.0,
        y[4] - 12.0,
        y[5] - 4.0,
        y[6] - 162.0,
    ]
}

fn tp067_state(x: [f64; 3]) -> [f64; 7] {
    let rx = 1.0 / x[0];
    let mut y2 = 1.6 * x[0];
    for _ in 0..20 {
        let y3 = 1.22 * y2 - x[0];
        let y6 = (x[1] + y3) * rx;
        let v2 = (112.0 + (13.167 - 0.6667 * y6) * y6) * 0.01;
        y2 = x[0] * v2;
    }

    let y3 = 1.22 * y2 - x[0];
    let y6 = (x[1] + y3) * rx;
    let mut y4 = 93.0;
    for _ in 0..20 {
        let y5 = 86.35 + 1.098 * y6 - 0.038 * y6.powi(2) + 0.325 * (y4 - 89.0);
        let y8 = -133.0 + 3.0 * y5;
        let y7 = 35.82 - 0.222 * y8;
        y4 = 9.8e4 * x[2] / (y2 * y7 + 1.0e3 * x[2]);
    }

    let y5 = 86.35 + 1.098 * y6 - 0.038 * y6.powi(2) + 0.325 * (y4 - 89.0);
    let y8 = -133.0 + 3.0 * y5;
    let y7 = 35.82 - 0.222 * y8;
    [y2, y3, y4, y5, y6, y7, y8]
}
