SetFactory("OpenCASCADE");
Disk(1) = {0, 0, 0, 100, 100};
Disk(2) = {0, 0, 0, 1, 1};
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };
Curve Loop(3) = {1};
Curve Loop(4) = {2};
Physical Surface(1) = {3};
