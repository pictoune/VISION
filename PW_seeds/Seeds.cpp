// Imagine++ project
// Project:  Seeds
// Author:   Pascal Monasse
// Student: TODO

#include <Imagine/Images.h>
#include <queue>
#include <string>
#include <iostream>
using namespace Imagine;
using namespace std;

// Default data
const char *DEF_im1=srcPath("im1.jpg"), *DEF_im2=srcPath("im2.jpg");
static int dmin=-30, dmax=-7; // Min and max disparities

/// Min NCC for a seed
static const float nccSeed=0.95f;

/// Radius of patch for correlation
static const int win=(9-1)/2;
/// To avoid division by 0 for constant patch
static const float EPS=0.1f;

/// A seed
struct Seed {
    Seed(int x0, int y0, int d0, float ncc0)
    : x(x0), y(y0), d(d0), ncc(ncc0) {}
    int x,y, d;
    float ncc;
};

/// Order by NCC
bool operator<(const Seed& s1, const Seed& s2) {
    return (s1.ncc<s2.ncc);
}

/// 4-neighbors
static const int dx[]={+1,  0, -1,  0};
static const int dy[]={ 0, -1,  0, +1};

/// Display disparity map
static Image<Color> displayDisp(const Image<int>& disp, Window W, int subW) {
    Image<Color> im(disp.width(), disp.height());
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++) {
            if(disp(i,j)<dmin || disp(i,j)>dmax)
                im(i,j) = CYAN;
            else {
                int g = 255*(disp(i,j)-dmin)/(dmax-dmin);
                im(i,j)= Color(g,g,g);
            }
        }
    setActiveWindow(W,subW);
    display(im);
    showWindow(W,subW);
    return im;
}

/// Show 3D window
static void show3D(const Image<Color>& im, const Image<int>& disp) {
#ifdef IMAGINE_OPENGL // Imagine++ must have been built with OpenGL support...
    // Intrinsic parameters given by Middlebury website
    const float f=3740;
    const float d0=-200; // Doll images cropped by this amount
    const float zoom=2; // Half-size images, should double measured disparity
    const float B=0.160; // Baseline in m
    FMatrix<float,3,3> K(0.0f);
    K(0,0)=-f/zoom; K(0,2)=disp.width()/2;
    K(1,1)= f/zoom; K(1,2)=disp.height()/2;
    K(2,2)=1.0f;
    K = inverse(K);
    K /= K(2,2);
    std::vector<FloatPoint3> pts;
    std::vector<Color> col;
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++)
            if(dmin<=disp(i,j) && disp(i,j)<=dmax) {
                float z = B*f/(zoom*disp(i,j)+d0);
                FloatPoint3 pt((float)i,(float)j,1.0f);
                pts.push_back(K*pt*z);
                col.push_back(im(i,j));
            }
    Mesh mesh(&pts[0], pts.size(), 0,0,0,0,VERTEX_COLOR);
    mesh.setColors(VERTEX, &col[0]);
    Window W = openWindow3D(512,512,"3D");
    setActiveWindow(W);
    showMesh(mesh);
#else
    std::cout << "No 3D: Imagine++ not built with OpenGL support" << std::endl;
#endif
}

/// Correlation between patches centered on (i1,j1) and (i2,j2). The values
/// m1 or m2 are subtracted from each pixel value.
static float correl(const Image<byte>& im1, int i1,int j1,float m1,
                    const Image<byte>& im2, int i2,int j2,float m2) {
    float dist=0.0f;

    float var_im1=0.0f, 
          var_im2=0.0f;
    
    for (int x = -win; x <= win; x++)
        for(int  y= -win; y <= win; y++){
            var_im1 += pow(im1(i1+x, j1+y) - m1, 2); // m1 = mean of pixel intensities
            var_im2 += pow(im2(i2+x, j2+y) - m2, 2); // m2 = mean of pixel intensities

            dist += (im1(i1+x,j1+y) - m1) * (im2(i2+x,j2+y) - m2);
        }

    float norm = sqrt(var_im1) * sqrt(var_im2) + EPS;

    // returns Normalized Cross-Correlation 
    return dist/norm;
}

/// Sum of pixel values in patch centered on (i,j).
static float sum(const Image<byte>& im, int i, int j) {
    float s=0.0f;

    for(int x = -win; x <= win; x++){
        for(int y = -win; y <= win; y++){
            s += im(i+x,j+y);
        }
    }

    return s;
}

/// Centered correlation of patches of size 2*win+1.
static float ccorrel(const Image<byte>& im1,int i1,int j1,
                     const Image<byte>& im2,int i2,int j2) {
    float m1 = sum(im1,i1,j1);
    float m2 = sum(im2,i2,j2);
    int w = 2*win+1;
    return correl(im1,i1,j1,m1/(w*w), im2,i2,j2,m2/(w*w));
}

/// Compute disparity map from im1 to im2, but only at points where NCC is
/// above nccSeed. Set to true the seeds and put them in Q.
static void find_seeds(Image<byte> im1, Image<byte> im2,
                       float nccSeed,
                       Image<int>& disp, Image<bool>& seeds,
                       std::priority_queue<Seed>& Q) {
    disp.fill(dmin-1);
    seeds.fill(false);
    while(! Q.empty())
        Q.pop();

    const int maxy = std::min(im1.height(),im2.height());
    const int refreshStep = (maxy-2*win)*5/100;

    bool reliable_pair = false;

    // finds the points for which the estimated disparity of two corresponding patches is sufficiently reliable
    for(int y=win; y+win<maxy; y++) {
        if((y-win-1)/refreshStep != (y-win)/refreshStep)
            std::cout << "Seeds: " << 5*(y-win)/refreshStep <<"%\r"<<std::flush;
        for(int x=win; x+win<im1.width(); x++) {
            float best_corr = -1.0f, corr;

            reliable_pair = false;

            for(int disp_ = dmin; disp_ <= dmax; disp_++){
                // ignores windows that are not fully in image
                if(((x + disp_) - win >= 0) && ((x + disp_) + win < im2.width())){
                    // computes the Normalized Cross-Correlation between the two patches
                    corr = ccorrel(im1, x, y, im2, x+disp_, y);

                    // if the two patches are probably those of two identical feature points
                    if(corr > 0.95 && corr > best_corr){
                        disp(x,y) = (float) disp_;
                        best_corr = corr; 

                        reliable_pair = true;
                    }
                }
            }

            if(reliable_pair){
                seeds(x,y) = true;
                Q.push(Seed(x, y, disp(x,y), best_corr));
            }
        }
    }
    std::cout << std::endl;

    /* Here we could also sort seeds of the queue from the largest correlation (front of the queue) to the smallest (back of the queue)
       but the program already takes a lot of time to execute */
}

/// Propagate seeds
static void propagate(Image<byte> im1, Image<byte> im2,
                      Image<int>& disp, Image<bool>& seeds,
                      std::priority_queue<Seed>& Q) {

    const int maxy = std::min(im1.height(),im2.height());
    int disp_;
    float corr, best_corr;

    while(! Q.empty()) {
        Seed s=Q.top();
        Q.pop();
        // for each of the neighbors of the feature point
        for(int i=0; i<4; i++) {
            int x=s.x+dx[i], y=s.y+dy[i];
            if(0<=x-win && x+win<im1.width() && 0<=y-win && y+win<maxy &&
               ! seeds(x,y)) {
                best_corr = -1.0f;

                // we deduce the disparity
                for(int delta_disp = -1; delta_disp < 1; delta_disp++){
                    disp_ = s.d + delta_disp; // disparity = disparity of the neighbor (+ small variation)

                    corr = ccorrel(im1, x, y, im2, x + disp_, y);

                    if(corr > best_corr){
                        disp(x,y) = disp_;
                        best_corr = corr;
                    }
                }

                seeds(x,y) = true;
                
                /* it's very unlikely that the real disparity is out of the interval [dMin,dMax]
                   so we push the seed only if it is not the case */
                if(dmin >= disp(x,y) && disp(x,y) <= dmax)
                    Q.push(Seed(x, y , disp(x,y), best_corr));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if(argc!=1 && argc!=5) {
        cerr << "Usage: " << argv[0] << " im1 im2 dmin dmax" << endl;
        return 1;
    }
    const char *im1=DEF_im1, *im2=DEF_im2;
    if(argc>1) {
        im1 = argv[1]; im2=argv[2]; dmin=stoi(argv[3]); dmax=stoi(argv[4]);
    }
    // Load and display images
    Image<Color> I1, I2;
    if(!load(I1,im1) || !load(I2,im2)) {
        cerr<< "Error loading image files" << endl;
        return 1;
    }
    std::string names[5]={"image 1","image 2","dense","seeds","propagation"};
    Window W = openComplexWindow(I1.width(), I1.height(), "Seeds propagation",
                                 5, names);
    setActiveWindow(W,0);
    display(I1,0,0);
    setActiveWindow(W,1);
    display(I2,0,0);

    Image<int> disp(I1.width(), I1.height());
    Image<bool> seeds(I1.width(), I1.height());
    std::priority_queue<Seed> Q;

    // Dense disparity
    find_seeds(I1, I2, -1.0f, disp, seeds, Q);
    save(displayDisp(disp,W,2), srcPath("0dense.png"));

    // Only seeds
    find_seeds(I1, I2, nccSeed, disp, seeds, Q);
    save(displayDisp(disp,W,3), srcPath("1seeds.png"));

    // Propagation of seeds
    propagate(I1, I2, disp, seeds, Q);
    save(displayDisp(disp,W,4), srcPath("2final.png"));

    // Show 3D (use shift click to animate)
    show3D(I1,disp);

    endGraphics();
    return 0;
}
