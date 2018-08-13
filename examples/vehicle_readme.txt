The vehicle.script is a use case for the prediction of a transport means

4 inputs:
age distance city/country-side(0/1) number_of_childrens
4 outputs:
bike bus moto car

Training set: vehicle.txt (values are scaled in [0,1])

name       age   distance city/country-side(0/1) number_of_childrens     bike bus moto car
user1      0.09   0.05            1                       0                1   0    0   0
user2      0.16   0.10            1                       0                1   0    0   0
user3      0.11   0.03            0                       0                1   0    0   0
user4      0.12   0.08            0                       0                1   0    0   0
user5      0.21   0.02            0                      0.1               1   0    0   0
user6      0.10   0.17            1                       0                0   1    0   0
user7      0.18   0.23            1                       0                0   1    0   0
user8      0.26   0.02            0                      0.1               0   1    0   0
user9      0.35   0.06            0                      0.2               0   1    0   0
user10     0.42   0.02            0                      0.2               0   1    0   0
user11     0.18   0.12            1                       0                0   0    1   0
user12     0.25   0.24            1                       0                0   0    1   0
user13     0.26   0.15            1                       0                0   0    1   0
user14     0.29   0.36            1                       0                0   0    1   0
user15     0.36   0.02            0                      0.1               0   0    1   0
user16     0.23   0.12            0                      0.1               0   0    0   1
user17     0.46   0.23            0                      0.3               0   0    0   1
user18     0.39   0.56            1                      0.4               0   0    0   1
user19     0.55   0.99            1                      0.3               0   0    0   1
user20     0.65   0.89            1                      0.5               0   0    0   1

