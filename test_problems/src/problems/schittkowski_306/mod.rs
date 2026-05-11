use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, Pair, ProblemCase, TypedProblemData, VecN, exact_solution_validation,
    make_typed_case, objective_validation, symbolic_compile,
};
use crate::manifest::ProblemTestSet;

// Upstream `PROB.FOR` is sparse: after TP119 it jumps directly to TP201.
// There are no TP120-TP200 subroutines in the public 2011 Fortran package.
const SOURCE: &str = "schittkowski_306";
const X_TOL: f64 = 1e-5;
const OBJECTIVE_TOL: f64 = 1e-8;
const PRIMAL_TOL: f64 = 1e-7;
const DUAL_TOL: f64 = 1e-5;
const COMPLEMENTARITY_TOL: f64 = 1e-6;

mod helpers;
mod tp001;
mod tp002;
mod tp003;
mod tp004;
mod tp005;
mod tp006;
mod tp007;
mod tp008;
mod tp009;
mod tp010;
mod tp011;
mod tp012;
mod tp013;
mod tp014;
mod tp015;
mod tp016;
mod tp017;
mod tp018;
mod tp019;
mod tp020;
mod tp021;
mod tp022;
mod tp023;
mod tp024;
mod tp025;
mod tp026;
mod tp027;
mod tp028;
mod tp029;
mod tp030;
mod tp031;
mod tp032;
mod tp033;
mod tp034;
mod tp035;
mod tp036;
mod tp037;
mod tp038;
mod tp039;
mod tp040;
mod tp041;
mod tp042;
mod tp043;
mod tp044;
mod tp045;
mod tp046;
mod tp047;
mod tp048;
mod tp049;
mod tp050;
mod tp051;
mod tp052;
mod tp053;
mod tp054;
mod tp055;
mod tp056;
mod tp057;
mod tp058;
mod tp059;
mod tp060;
mod tp061;
mod tp062;
mod tp063;
mod tp064;
mod tp065;
mod tp066;
mod tp067;
mod tp068;
mod tp069;
mod tp070;
mod tp071;
mod tp072;
mod tp073;
mod tp074;
mod tp075;
mod tp076;
mod tp077;
mod tp078;
mod tp079;
mod tp080;
mod tp081;
mod tp083;
mod tp084;
mod tp085;
mod tp086;
mod tp087;
mod tp088;
mod tp089;
mod tp090;
mod tp091;
mod tp092;
mod tp093;
mod tp095;
mod tp096;
mod tp097;
mod tp098;
mod tp099;
mod tp100;
mod tp101;
mod tp102;
mod tp103;
mod tp104;
mod tp105;
mod tp106;
mod tp107;
mod tp108;
mod tp109;
mod tp110;
mod tp111;
mod tp112;
mod tp113;
mod tp114;
mod tp116;
mod tp117;
mod tp118;
mod tp119;
mod tp201;
mod tp202;
mod tp203;
mod tp204;
mod tp205;
mod tp206;
mod tp207;
mod tp208;
mod tp209;
mod tp210;
mod tp211;
mod tp212;
mod tp213;
mod tp214;
mod tp215;
mod tp216;
mod tp217;
mod tp218;
mod tp219;
mod tp220;
mod tp221;
mod tp222;
mod tp223;
mod tp224;
mod tp225;
mod tp226;
mod tp227;
mod tp228;
mod tp229;
mod tp230;
mod tp231;
mod tp232;
mod tp233;
mod tp234;
mod tp235;
mod tp236;
mod tp237;
mod tp238;
mod tp239;
mod tp240;
mod tp241;
mod tp242;
mod tp243;
mod tp244;
mod tp245;
mod tp246;
mod tp247;
mod tp248;
mod tp249;
mod tp250;
mod tp251;
mod tp252;
mod tp253;
mod tp254;
mod tp255;
mod tp256;
mod tp257;
mod tp258;
mod tp259;
mod tp260;
mod tp261;
mod tp262;
mod tp263;
mod tp264;
mod tp265;
mod tp266;
mod tp267;
mod tp268;
mod tp269;
mod tp270;
mod tp271;
mod tp272;
mod tp273;
mod tp274;
mod tp275;
mod tp276;
mod tp277;
mod tp278;
mod tp279;
mod tp280;
mod tp281;
mod tp282;
mod tp283;
mod tp284;
mod tp285;
mod tp286;
mod tp287;
mod tp288;
mod tp289;
mod tp290;
mod tp291;
mod tp292;
mod tp293;
mod tp294;
mod tp295;
mod tp296;
mod tp297;
mod tp298;
mod tp299;
mod tp300;
mod tp301;
mod tp302;
mod tp303;
mod tp304;
mod tp305;
mod tp306;

#[cfg(test)]
mod tests;

use tp001::tp001;
use tp002::tp002;
use tp003::tp003;
use tp004::tp004;
use tp005::tp005;
use tp006::tp006;
use tp007::tp007;
use tp008::tp008;
use tp009::tp009;
use tp010::tp010;
use tp011::tp011;
use tp012::tp012;
use tp013::tp013;
use tp014::tp014;
use tp015::tp015;
use tp016::tp016;
use tp017::tp017;
use tp018::tp018;
use tp019::tp019;
use tp020::tp020;
use tp021::tp021;
use tp022::tp022;
use tp023::tp023;
use tp024::tp024;
use tp025::tp025;
use tp026::tp026;
use tp027::tp027;
use tp028::tp028;
use tp029::tp029;
use tp030::tp030;
use tp031::tp031;
use tp032::tp032;
use tp033::tp033;
use tp034::tp034;
use tp035::tp035;
use tp036::tp036;
use tp037::tp037;
use tp038::tp038;
use tp039::tp039;
use tp040::tp040;
use tp041::tp041;
use tp042::tp042;
use tp043::tp043;
use tp044::tp044;
use tp045::tp045;
use tp046::tp046;
use tp047::tp047;
use tp048::tp048;
use tp049::tp049;
use tp050::tp050;
use tp051::tp051;
use tp052::tp052;
use tp053::tp053;
use tp054::tp054;
use tp055::tp055;
use tp056::tp056;
use tp057::tp057;
use tp058::tp058;
use tp059::tp059;
use tp060::tp060;
use tp061::tp061;
use tp062::tp062;
use tp063::tp063;
use tp064::tp064;
use tp065::tp065;
use tp066::tp066;
use tp067::tp067;
use tp068::tp068;
use tp069::tp069;
use tp070::tp070;
use tp071::tp071;
use tp072::tp072;
use tp073::tp073;
use tp074::tp074;
use tp075::tp075;
use tp076::tp076;
use tp077::tp077;
use tp078::tp078;
use tp079::tp079;
use tp080::tp080;
use tp081::tp081;
use tp083::tp083;
use tp084::tp084;
use tp085::tp085;
use tp086::tp086;
use tp087::tp087;
use tp088::tp088;
use tp089::tp089;
use tp090::tp090;
use tp091::tp091;
use tp092::tp092;
use tp093::tp093;
use tp095::tp095;
use tp096::tp096;
use tp097::tp097;
use tp098::tp098;
use tp099::tp099;
use tp100::tp100;
use tp101::tp101;
use tp102::tp102;
use tp103::tp103;
use tp104::tp104;
use tp105::tp105;
use tp106::tp106;
use tp107::tp107;
use tp108::tp108;
use tp109::tp109;
use tp110::tp110;
use tp111::tp111;
use tp112::tp112;
use tp113::tp113;
use tp114::tp114;
use tp116::tp116;
use tp117::tp117;
use tp118::tp118;
use tp119::tp119;
use tp201::tp201;
use tp202::tp202;
use tp203::tp203;
use tp204::tp204;
use tp205::tp205;
use tp206::tp206;
use tp207::tp207;
use tp208::tp208;
use tp209::tp209;
use tp210::tp210;
use tp211::tp211;
use tp212::tp212;
use tp213::tp213;
use tp214::tp214;
use tp215::tp215;
use tp216::tp216;
use tp217::tp217;
use tp218::tp218;
use tp219::tp219;
use tp220::tp220;
use tp221::tp221;
use tp222::tp222;
use tp223::tp223;
use tp224::tp224;
use tp225::tp225;
use tp226::tp226;
use tp227::tp227;
use tp228::tp228;
use tp229::tp229;
use tp230::tp230;
use tp231::tp231;
use tp232::tp232;
use tp233::tp233;
use tp234::tp234;
use tp235::tp235;
use tp236::tp236;
use tp237::tp237;
use tp238::tp238;
use tp239::tp239;
use tp240::tp240;
use tp241::tp241;
use tp242::tp242;
use tp243::tp243;
use tp244::tp244;
use tp245::tp245;
use tp246::tp246;
use tp247::tp247;
use tp248::tp248;
use tp249::tp249;
use tp250::tp250;
use tp251::tp251;
use tp252::tp252;
use tp253::tp253;
use tp254::tp254;
use tp255::tp255;
use tp256::tp256;
use tp257::tp257;
use tp258::tp258;
use tp259::tp259;
use tp260::tp260;
use tp261::tp261;
use tp262::tp262;
use tp263::tp263;
use tp264::tp264;
use tp265::tp265;
use tp266::tp266;
use tp267::tp267;
use tp268::tp268;
use tp269::tp269;
use tp270::tp270;
use tp271::tp271;
use tp272::tp272;
use tp273::tp273;
use tp274::tp274;
use tp275::tp275;
use tp276::tp276;
use tp277::tp277;
use tp278::tp278;
use tp279::tp279;
use tp280::tp280;
use tp281::tp281;
use tp282::tp282;
use tp283::tp283;
use tp284::tp284;
use tp285::tp285;
use tp286::tp286;
use tp287::tp287;
use tp288::tp288;
use tp289::tp289;
use tp290::tp290;
use tp291::tp291;
use tp292::tp292;
use tp293::tp293;
use tp294::tp294;
use tp295::tp295;
use tp296::tp296;
use tp297::tp297;
use tp298::tp298;
use tp299::tp299;
use tp300::tp300;
use tp301::tp301;
use tp302::tp302;
use tp303::tp303;
use tp304::tp304;
use tp305::tp305;
use tp306::tp306;

pub(crate) fn cases() -> Vec<ProblemCase> {
    vec![
        tp001(),
        tp002(),
        tp003(),
        tp004(),
        tp005(),
        tp006(),
        tp007(),
        tp008(),
        tp009(),
        tp010(),
        tp011(),
        tp012(),
        tp013(),
        tp014(),
        tp015(),
        tp016(),
        tp017(),
        tp018(),
        tp019(),
        tp020(),
        tp021(),
        tp022(),
        tp023(),
        tp024(),
        tp025(),
        tp026(),
        tp027(),
        tp028(),
        tp029(),
        tp030(),
        tp031(),
        tp032(),
        tp033(),
        tp034(),
        tp035(),
        tp036(),
        tp037(),
        tp038(),
        tp039(),
        tp040(),
        tp041(),
        tp042(),
        tp043(),
        tp044(),
        tp045(),
        tp046(),
        tp047(),
        tp048(),
        tp049(),
        tp050(),
        tp051(),
        tp052(),
        tp053(),
        tp054(),
        tp055(),
        tp056(),
        tp057(),
        tp058(),
        tp059(),
        tp060(),
        tp061(),
        tp062(),
        tp063(),
        tp064(),
        tp065(),
        tp066(),
        tp067(),
        tp068(),
        tp069(),
        tp070(),
        tp071(),
        tp072(),
        tp073(),
        tp074(),
        tp075(),
        tp076(),
        tp077(),
        tp078(),
        tp079(),
        tp080(),
        tp081(),
        tp083(),
        tp084(),
        tp085(),
        tp086(),
        tp087(),
        tp088(),
        tp089(),
        tp090(),
        tp091(),
        tp092(),
        tp093(),
        tp095(),
        tp096(),
        tp097(),
        tp098(),
        tp099(),
        tp100(),
        tp101(),
        tp102(),
        tp103(),
        tp104(),
        tp105(),
        tp106(),
        tp107(),
        tp108(),
        tp109(),
        tp110(),
        tp111(),
        tp112(),
        tp113(),
        tp114(),
        tp116(),
        tp117(),
        tp118(),
        tp119(),
        tp201(),
        tp202(),
        tp203(),
        tp204(),
        tp205(),
        tp206(),
        tp207(),
        tp208(),
        tp209(),
        tp210(),
        tp211(),
        tp212(),
        tp213(),
        tp214(),
        tp215(),
        tp216(),
        tp217(),
        tp218(),
        tp219(),
        tp220(),
        tp221(),
        tp222(),
        tp223(),
        tp224(),
        tp225(),
        tp226(),
        tp227(),
        tp228(),
        tp229(),
        tp230(),
        tp231(),
        tp232(),
        tp233(),
        tp234(),
        tp235(),
        tp236(),
        tp237(),
        tp238(),
        tp239(),
        tp240(),
        tp241(),
        tp242(),
        tp243(),
        tp244(),
        tp245(),
        tp246(),
        tp247(),
        tp248(),
        tp249(),
        tp250(),
        tp251(),
        tp252(),
        tp253(),
        tp254(),
        tp255(),
        tp256(),
        tp257(),
        tp258(),
        tp259(),
        tp260(),
        tp261(),
        tp262(),
        tp263(),
        tp264(),
        tp265(),
        tp266(),
        tp267(),
        tp268(),
        tp269(),
        tp270(),
        tp271(),
        tp272(),
        tp273(),
        tp274(),
        tp275(),
        tp276(),
        tp277(),
        tp278(),
        tp279(),
        tp280(),
        tp281(),
        tp282(),
        tp283(),
        tp284(),
        tp285(),
        tp286(),
        tp287(),
        tp288(),
        tp289(),
        tp290(),
        tp291(),
        tp292(),
        tp293(),
        tp294(),
        tp295(),
        tp296(),
        tp297(),
        tp298(),
        tp299(),
        tp300(),
        tp301(),
        tp302(),
        tp303(),
        tp304(),
        tp305(),
        tp306(),
    ]
}
