#ifndef CONVEX_HULL_H
#define CONVEX_HULL_H

#include <algorithm>
#include <vector>


struct Point {
    float x, y;
    Point(int x, int y) {
        this->x = x;
        this->y = y;
    };
    Point() {
        this->x = 0;
        this->y = 0;
    };
};

bool compare(Point a, Point b);
bool clockwise(Point a, Point b, Point c);
bool counterClockwise(Point a, Point b, Point c);

void convexHull(std::vector<Point>& a);

#endif
