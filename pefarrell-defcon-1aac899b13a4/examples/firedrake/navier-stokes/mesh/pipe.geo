DefineConstant[ lc  = {0.75}];

// ---------------- Circle ----------------
Point(1) = {0, -1, 0, lc};
Point(2) = {2.5, -1, 0, lc};
Point(3) = {2.5, -6, 0, lc};
Point(4) = {150, -6, 0, lc};
Point(5) = {150, 6, 0, lc};
Point(6) = {2.5, 6, 0, lc};
Point(7) = {2.5, 1, 0, lc};
Point(8) = {0, 1, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};

//boundary and physical curves
Line Loop(9) = {1, 2, 3, 4, 5, 6, 7, 8};
Physical Line("Inflow", 10) = {8};
Physical Line("Outflow", 11) = {4};
Physical Line("WallFixed", 12) = {1, 2, 3, 5, 6, 7};

//domain and physical surface
Plane Surface(1) = {9};
Physical Surface("Pipe", 2) = {1};
