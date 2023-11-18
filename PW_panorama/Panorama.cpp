// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
#include <cstdlib>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
int getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- TODO/A completer ----------
    Window win;
    int sub_win;
    IntPoint2 cliked_pt;
    bool done = false;
    int nb_pts = 0;
    Color c;
    char r, g, b;

    while(!done){
        char button = anyGetMouse(cliked_pt, win, sub_win);

        if(button == 3){
            done = true;
            return 0;
        }
        else{
            if((nb_pts % 2) == 0){
                r = rand() % 256; g = rand() % 256; b = rand() % 256;
                c = Color(r, g, b);
            }

            setActiveWindow(win);
            fillCircle(cliked_pt, 9, c, false);

            if(win == w1){
                pts1.push_back(cliked_pt);
                cout << "Active window: 1" << endl;
            }
            else if(win == w2){
                pts2.push_back(cliked_pt);
                cout << "Active window: 2" << endl;
            }
            cout << "Coordinates of the new point " << nb_pts << ": " << cliked_pt << endl;
            cout << endl;

            nb_pts += 1;
        }
    }
    return 0;
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2){
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------
    for(int eq = 0; eq < 2 * n; eq++){
        int pt_idx = eq / 2;

        IntPoint2 pt = pts1[pt_idx];
        int x = pt[0];
        int y = pt[1];

        IntPoint2 pt_prime = pts2[pt_idx];
        int x_prime = pt_prime[0];
        int y_prime = pt_prime[1];

        if(eq % 2 == 0){
            A(eq, 0) = x;
            A(eq, 1) = y;
            A(eq, 2) = 1;
            A(eq, 3) = 0;
            A(eq, 4) = 0;
            A(eq, 5) = 0;
            A(eq, 6) = -x_prime*x;
            A(eq, 7) = -x_prime*y;
        
            B[eq] = x_prime;
        }
        else if(eq % 2 == 1){
            A(eq, 0) = 0;
            A(eq, 1) = 0;
            A(eq, 2) = 0;
            A(eq, 3) = x;
            A(eq, 4) = y;
            A(eq, 5) = 1;
            A(eq, 6) = -y_prime*x;
            A(eq, 7) = -y_prime*y;
            
            B[eq] = y_prime;
        }
    }
    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);
    // ------------- TODO/A completer ----------
    Matrix<float> inv_H = inverse(H);
    Vector<float> pt(3), pt_old(3);

    for(int x = 0; x < I.width(); x++){
        for(int y = 0; y < I.height(); y++){
            // Bottom left corner
            pt[0] = x + x0; pt[1] = y + y0; pt[2] = 1;

            bool inside_I2 = (pt[0] > 0 && pt[1] > 0 && pt[0] < I2.width() && pt[1] < I2.height());

            if(inside_I2){
                I(x, y) = I2(pt[0], pt[1]);
            }
            pt_old = pt;

            // Position in the system of I1 = inversed_homography(position of the point in the system I2)
            pt = inv_H * pt; pt /= pt[2];
            bool inside_I1 = (pt[0] > 0 && pt[1] > 0 && pt[0] < I1.width() && pt[1] < I1.height());

            if(inside_I1){
                I(x, y) = I1.interpolate(pt[0], pt[1]);
            }

            bool overlap = (inside_I1 && inside_I2);

            if(overlap){
                // Average of the colors of the corresponding points
                I(x, y)[0] = (I2(pt_old[0], pt[1])[0] + I1.interpolate(pt[0], pt[1]).r()) / 2;
                I(x, y)[1] = (I2(pt_old[0], pt[1])[1] + I1.interpolate(pt[0], pt[1]).g()) / 2;
                I(x, y)[2] = (I2(pt_old[0], pt[1])[2] + I1.interpolate(pt[0], pt[1]).b()) / 2;
            }
        }
    }
    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;

    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);

    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
