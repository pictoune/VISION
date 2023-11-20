// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match
{
    float x1, y1, x2, y2;
};


// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color, 2> I1, Image<Color, 2> I2,
              vector<Match> &matches)
{
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0, 0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(), 0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0 * 100.0;
    for (size_t i = 0; i < feats1.size(); i++)
    {
        SIFTDetector::Feature f1 = feats1[i];
        for (size_t j = 0; j < feats2.size(); j++)
        {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if (d < MAX_DISTANCE)
            {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}


vector<Match> sample_matches(vector<Match> &matches, int k)
{
    int n = matches.size();

    vector<Match> sampled_matches;
    int samples_indices[k], nb_sampled_matches = 0;

    bool already_sampled;

    while (nb_sampled_matches < k)
    {
        bool sampled = false;
        int sample_idx;

        while (!sampled)
        {
            already_sampled = false;
            sample_idx = rand() % n;

            for (int j = 0; j < nb_sampled_matches; j++)
                if (sample_idx == samples_indices[j])
                {
                    already_sampled = true;
                    break;
                }
            if (!already_sampled)
                sampled = true;
        }

        samples_indices[nb_sampled_matches] = sample_idx;
        sampled_matches.push_back(matches[sample_idx]);

        nb_sampled_matches += 1;
    }
    return sampled_matches;
}


// estimates F from 8 matches
FMatrix<float, 3, 3> eight_pt_algo(vector<Match> &sampled_matches, bool refinement)
{
    FMatrix<float, 3, 3> N;
    N(0, 0) = N(1, 1) = 0.001;
    N(2, 2) = 1;

    FloatPoint3 pt_1, pt_2;
    FMatrix<float, 9, 9> A;

    char nb_eq = 8 + int(refinement);
    for (int i = 0; i < nb_eq; i++)
    {
        pt_1[0] = sampled_matches[i].x1;
        pt_1[1] = sampled_matches[i].y1;
        pt_1[2] = 1;
        pt_2[0] = sampled_matches[i].x2;
        pt_2[1] = sampled_matches[i].y2;
        pt_2[2] = 1;

        // normalization
        pt_1 = N * pt_1;
        pt_2 = N * pt_2;

        float x1 = pt_1[0];
        float y1 = pt_1[1];
        float x2 = pt_2[0];
        float y2 = pt_2[1];

        A(i, 0) = x1 * x2;
        A(i, 1) = x1 * y2;
        A(i, 2) = x1;
        A(i, 3) = y1 * x2;
        A(i, 4) = y1 * y2;
        A(i, 5) = y1;
        A(i, 6) = x2;
        A(i, 7) = y2;
        A(i, 8) = 1;
    }

    if (!refinement)
    {
        for (int col = 0; col < 9; col++)
        {
            A(8, col) = 0; // we add another "equation" because it's easier to use SVD with a square matrix
        }
    }

    FMatrix<float, 9, 9> U;
    FVector<float, 9> S;
    FMatrix<float, 9, 9> Vt;

    svd(A, U, S, Vt);

    // F is the eigenvector associated to the smallest eighenvalue
    FVector<float, 9> last_eig_vec = Vt.getRow(8);

    FMatrix<float, 3, 3> F;

    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            F(row, col) = last_eig_vec[3 * row + col];
        }
    }
    // To enforce the rank-2 constraint we keep only the two eigenvectors corresponding to the two biggest eigenvalues
    FVector<float, 3> S2;
    FMatrix<float, 3, 3> U2, Vt2;

    svd(F, U2, S2, Vt2);

    S2[2] = 0;
    F = U2 * Diagonal(S2) * Vt2;

    F = transpose(N) * F * N;

    return F;
}


int update_IterNb(int m, int n, int k)
{
    float denominator = log(1 - pow((float)m / (float)n, k));

    if (denominator < 0)
    {
        return ceil(log(BETA) / denominator);
    }
    else
    {
        return -1;
    }
}


vector<int> find_inliers(vector<Match> matches, FMatrix<float, 3, 3> F, const float distMax)
{
    vector<int> inliers;
    FloatPoint3 pt_1, pt_2;
    FVector<float, 3> pt;

    FMatrix<float, 3, 3> F_T = transpose(F);

    for (size_t i = 0; i < matches.size(); i++)
    {
        pt_1[0] = matches[i].x1;
        pt_1[1] = matches[i].y1;
        pt_1[2] = 1;
        pt_2[0] = matches[i].x2;
        pt_2[1] = matches[i].y2;
        pt_2[2] = 1;

        pt[0] = matches[i].x1;
        pt[1] = matches[i].y1;
        pt[2] = 1;

        float d = abs(((F_T * pt)[0] * matches[i].x2) + ((F_T * pt)[1] * matches[i].y2) + (F_T * pt)[2]) / sqrt(pow((F_T * pt)[0], 2) + pow((F_T * pt)[1], 2));

        if (d < distMax)
        {
            inliers.push_back(i);
        }
    }
    return inliers;
}


// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float, 3, 3> computeF(vector<Match> &matches)
{
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    double Niter = 100000;      // Adjusted dynamically
    FMatrix<float, 3, 3> bestF, F;
    vector<int> bestInliers;
    vector<Match> sampled_matches;
    int nb_sampled_matches = 0, k = 8;

    while (nb_sampled_matches < Niter)
    {
        sampled_matches = sample_matches(matches, k);

        F = eight_pt_algo(sampled_matches, false);

        // finds inliers
        vector<int> inliers = find_inliers(matches, F, distMax);

        bool improvement = (inliers.size() > bestInliers.size());

        if (improvement)
        {
            bestInliers = inliers;
            bestF = F;

            int res = update_IterNb(bestInliers.size(), matches.size(), k);

            // negative number of iteration
            if (res != -1)
            {
                Niter = res;
            }
        }
        nb_sampled_matches += 1;
    }
    // Updating matches with inliers only
    vector<Match> all = matches;
    matches.clear();

    for (size_t i = 0; i < bestInliers.size(); i++)
    {
        matches.push_back(all[bestInliers[i]]);

        if (i == (size_t)k)
        {
            // refines resulting F with k inliers
            // bestF = eight_pt_algo(matches,true); // DOESN'T WORK !!!
        }
    }
    return bestF;
}


// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, const FMatrix<float, 3, 3> &F)
{
    while (true)
    {
        int x, y;
        if (getMouse(x, y) == 3)
            break;

        Color c = Color(rand() % 256,
                        rand() % 256,
                        rand() % 256);

        // shows clicked point
        fillCircle(x, y, 5, c, false);

        DoublePoint3 pt;
        pt[0] = x;
        pt[1] = y;
        pt[2] = 1;

        FVector<float, 3> epi_line;

        IntPoint2 left_pt, right_pt;

        int I1_width = I1.width();

        // the user clicked in the left image
        if (x <= I1_width)
        {
            epi_line = transpose(F) * pt;

            left_pt[0] = I1_width;
            right_pt[0] = I1_width * 2;
        }
        // the user clicked in the right image
        else if (x > I1_width)
        {
            pt[0] -= I1_width;
            epi_line = F * pt;

            left_pt[0] = 0;
            right_pt[0] = I1_width;
        }
        left_pt[1] = -(epi_line[2]) / epi_line[1];
        right_pt[1] = -(epi_line[2] + epi_line[0] * I1_width) / epi_line[1];

        // shows associated epipolar line in other image
        drawLine(left_pt, right_pt, c, 3);
    }
}


int main(int argc, char *argv[])
{
    srand((unsigned int)time(0));

    const char *s1 = argc > 1 ? argv[1] : srcPath("im1.jpg");
    const char *s2 = argc > 2 ? argv[2] : srcPath("im2.jpg");

    // Load and display images
    Image<Color, 2> I1, I2;
    if (!load(I1, s1) ||
        !load(I2, s2))
    {
        cerr << "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2 * w, I1.height());
    display(I1, 0, 0);
    display(I2, w, 0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100, 20, std::to_string(n) + " matches", RED);
    click();

    FMatrix<float, 3, 3> F = computeF(matches);
    cout << "F=" << endl
         << F;

    // Redisplay with matches
    display(I1, 0, 0);
    display(I2, w, 0);
    for (size_t i = 0; i < matches.size(); i++)
    {
        Color c(rand() % 256, rand() % 256, rand() % 256);
        fillCircle(matches[i].x1 + 0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2 + w, matches[i].y2, 2, c);
    }
    drawString(100, 20, to_string(matches.size()) + "/" + to_string(n) + " inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1, 0, 0);
    display(I2, w, 0);
    displayEpipolar(I1, F);

    endGraphics();
    return 0;
}
