#include "fwFit_MixedLS_1r2star.h"

double complexPhase(double x, double y) {
  double curPhi = 0.0; 


  if(x>0) {
    curPhi = atan(y/x);
  } else if(x<0) {
    if(y>=0) {
      curPhi = atan(y/x) + PI;
    } else {
      curPhi = atan(y/x) - PI;
    }
    
  } else if(x==0) {
    
    if(y>=0) {
      curPhi = PI/2;
    } else {
      curPhi = -PI/2;
    }
    
  }


  return curPhi;
}

// Implement an objective functor for Eigen.
struct ParabolicError_Mixed
{
    constexpr static bool ComputesJacobian = true;
    data_str *data;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval,
                    Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
    {


        double shat, shatr, shati, CS, SN, EXP;;
        double curJ1,curJ2,curJ3;
        double curJ4,curJ5,curJ6;
        double expr2, sinfm, cosfm;

        int nte = data->nte;
        double *cursr = data->cursr;
        double *cursi = data->cursi;
        double *te = data->te;
        double *swr = data->swr;
        double *swi = data->swi;
        double *sfr = data->sfr;
        double *sfi = data->sfi;


        double W = xval(0);
        double F = xval(1);
        double phi = xval(2);
        double r2 = xval(3);
        double fieldmap = xval(4);


        int NUM_MAGN = 1;

        fval.resize(2*nte-NUM_MAGN);
        for(lsqcpp::Index kt = 0; kt < NUM_MAGN; ++kt)
        {
            EXP = exp(-te[kt]*r2);
            shat = EXP*sqrt((W*swr[kt] + F*sfr[kt])*(W*swr[kt] + F*sfr[kt]) + (W*swi[kt] + F*sfi[kt])*(W*swi[kt] + F*sfi[kt]));
            fval(kt) = shat - sqrt(cursr[kt]*cursr[kt] + cursi[kt]*cursi[kt]);
        }  
        for(lsqcpp::Index kt = NUM_MAGN; kt < nte; ++kt)
        {
            CS = cos(phi + 2*PI*fieldmap*te[kt]);
            SN = sin(phi + 2*PI*fieldmap*te[kt]);
            EXP = exp(-te[kt]*r2);

            shatr = CS*EXP*(W*swr[kt] + F*sfr[kt]) - SN*EXP*(W*swi[kt] + F*sfi[kt]);
            shati = SN*EXP*(W*swr[kt] + F*sfr[kt]) + CS*EXP*(W*swi[kt] + F*sfi[kt]);

            fval(kt) = shatr - cursr[kt];
            fval(kt+nte-NUM_MAGN) = shati - cursi[kt];
        }  



        // calculate the jacobian explicitly
        jacobian.setZero(2*nte-NUM_MAGN, xval.size());
        for(lsqcpp::Index kt = 0; kt < NUM_MAGN; ++kt)
        {

            EXP = exp(-te[kt]*r2);

            shat = sqrt((W*swr[kt] + F*sfr[kt])*(W*swr[kt] + F*sfr[kt]) + (W*swi[kt] + F*sfi[kt])*(W*swi[kt] + F*sfi[kt])) + 1e-12;

            curJ1 = EXP*(W*swr[kt] + F*sfr[kt])/shat;
            jacobian(kt,0) = curJ1;

            curJ2 = EXP*(F*(sfr[kt]*sfr[kt] + sfi[kt]*sfi[kt]) + W*sfr[kt])/shat;
            jacobian(kt,1) = curJ2;
            
            jacobian(kt,2) = 0.0;

            curJ4 = EXP*(-W*W*te[kt] - W*F*sfr[kt]*te[kt] -F*F*te[kt]*(sfr[kt]*sfr[kt] + sfi[kt]*sfi[kt]) - W*F*sfr[kt]*te[kt])/shat;
            jacobian(kt,3) = curJ4;

            jacobian(kt,4) = 0.0;

        }
        for(lsqcpp::Index kt = NUM_MAGN; kt < nte; ++kt)
        {

            expr2 = exp(-te[kt]*r2);
            sinfm = sin(phi + 2*PI*fieldmap*te[kt]);
            cosfm = cos(phi + 2*PI*fieldmap*te[kt]);

            shatr=cosfm*expr2*(W*swr[kt] + F*sfr[kt]) - sinfm*expr2*(W*swi[kt] + F*sfi[kt]);
            shati=sinfm*expr2*(W*swr[kt] + F*sfr[kt]) + cosfm*expr2*(W*swi[kt] + F*sfi[kt]);

            curJ1 = cosfm*expr2*swr[kt] - sinfm*expr2*swi[kt];
            jacobian(kt,0) = curJ1;
            curJ1 = sinfm*expr2*swr[kt] + cosfm*expr2*swi[kt];
            jacobian(kt+nte-NUM_MAGN,0) = curJ1;

            curJ2 = cosfm*expr2*sfr[kt] - sinfm*expr2*sfi[kt];
            jacobian(kt,1) = curJ2;
            curJ2 = sinfm*expr2*sfr[kt] + cosfm*expr2*sfi[kt];
            jacobian(kt+nte-NUM_MAGN,1) = curJ2;

            curJ3 = -sinfm*expr2*(W*swr[kt] + F*sfr[kt]) - cosfm*expr2*(W*swi[kt] + F*sfi[kt]);
            jacobian(kt,2) = curJ3;
            curJ3 = cosfm*expr2*(W*swr[kt] + F*sfr[kt]) - sinfm*expr2*(W*swi[kt] + F*sfi[kt]);
            jacobian(kt+nte-NUM_MAGN,2) = curJ3;

            curJ4 = -te[kt]*shatr;
            jacobian(kt,3) = curJ4;
            curJ4 = -te[kt]*shati;
            jacobian(kt+nte-NUM_MAGN,3) = curJ4;
            
            curJ5 = 2*PI*te[kt]*(-sinfm*expr2*(W*swr[kt] + F*sfr[kt]) - cosfm*expr2*(W*swi[kt] + F*sfi[kt]));
            jacobian(kt,4) = curJ5;
            curJ5 =  2*PI*te[kt]*(cosfm*expr2*(W*swr[kt] + F*sfr[kt]) - sinfm*expr2*(W*swi[kt] + F*sfi[kt]));
            jacobian(kt+nte-NUM_MAGN,4) = curJ5;


        }

        
    }
};



void fwFit_MixedLS_1r2star::initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in){

    this->imDataParams = imDataParams_in;
    this->algoParams = algoParams_in;
    this->initParams = initParams_in;

    this->nte = imDataParams->nte;
    this->fieldStrength = imDataParams->FieldStrength;
    this->clockwise = imDataParams->PrecessionIsClockwise;
    this->nx = imDataParams->im_dim[0];
    this->ny = imDataParams->im_dim[1];
    this->nf = std::max(nx,ny);

    cursr = new double[nte];
    cursi = new double[nte];
    sfr = new double[nte];
    sfi = new double[nte];
    swr = new double[nte];
    swi = new double[nte];
    //fre = new double[nte];
    //fim = new double[nte];
    te = new double[nte];

    /* Get algoParams */
    double waterAmp = algoParams->species_wat_amp[0];
    double *relAmps = algoParams->species_fat_amp;
    double *fPPM = algoParams->species_fat_freq;
    nf = algoParams->NUM_FAT_PEAKS;



    fF = new double[nf];

    for(int kf=0;kf<nf;kf++) {
        fF[kf] = fPPM[kf]*GYRO*fieldStrength;
    }

    /* Get initParams */
    initWr = initParams->water_r_init;
    initFr = initParams->fat_r_init; 
    initWi = initParams->water_i_init;
    initFi = initParams->fat_i_init;
    initR2 = initParams->r2s_init;
    initFieldmap = initParams->fm_init;
    masksignal = initParams->masksignal_init;


    for(int kf=0;kf<nte;kf++){
        te[kf] = imDataParams->TE[kf];
    }


    outR2 = new double[nx*ny];
    outFieldmap =  new double[nx*ny];
    outWr =  new double[nx*ny];
    outWi =  new double[nx*ny];
    outFr =  new double[nx*ny];
    outFi =  new double[nx*ny];

    /* Initialize water/fat signal models */
    fF = (double *)malloc(nf*sizeof(double));
    for(int kf=0;kf<nf;kf++) {
        fF[kf] = fPPM[kf]*GYRO*fieldStrength;
    }
    for(int kt=0;kt<nte;kt++) {
        swr[kt] = waterAmp;
        swi[kt] = 0.0;
        sfr[kt] = 0.0;
        sfi[kt] = 0.0;
        for(int kf=0;kf<nf;kf++) {
        
            sfr[kt] = sfr[kt] + relAmps[kf]*cos(2*PI*te[kt]*fF[kf]);
            sfi[kt] = sfi[kt] + relAmps[kf]*sin(2*PI*te[kt]*fF[kf]);

        }    

    }


}


void fwFit_MixedLS_1r2star::fit_all(){


    //printf("In fit_all()\n");

    lsqcpp::LevenbergMarquardtX<double, ParabolicError_Mixed> optimizer;
    // Set number of iterations as stop criterion.
    optimizer.setMaximumIterations(50);
    // Set the minimum length of the gradient.
    optimizer.setMinimumGradientLength(1e-4);
    // Set the minimum length of the step.
    optimizer.setMinimumStepLength(1e-4);
    // Set the minimum least squares error.
    optimizer.setMinimumError(0);
    // Set the parameters of the step method (Levenberg Marquardt).
    optimizer.setMethodParameters({1.0, 2.0, 0.5, 100});
    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbosity(0);



    //printf("Assign data structure\n");

    data_str data;
    data.nte = nte;
    data.cursr = cursr;
    data.cursi = cursi;
    data.te = te;
    data.swr = swr;
    data.swi = swi;
    data.sfr = sfr;
    data.sfi = sfi;

    ParabolicError_Mixed costFunction;
    costFunction.data = &data;
    optimizer.setObjective(costFunction);

    double *imsr = imDataParams->images_r;
    double *imsi = imDataParams->images_i;

    double curPhi, curAmpW, curAmpF;

    /* Loop over all pixels */
    for(int kx=0;kx<nx;kx++) {
        for(int ky=0;ky<ny;ky++) {

            if(masksignal[kx + ky*nx] > 0.1){

                /* Get signal at current voxel */
                if(clockwise>0) {
                    for(int kt=0;kt<nte;kt++) {
                        cursr[kt] = imsr[kx + ky*nx + kt*nx*ny];
                        cursi[kt] = imsi[kx + ky*nx + kt*nx*ny];
                    }
                } else {
                    for(int kt=0;kt<nte;kt++) {
                        cursr[kt] = imsr[kx + ky*nx + kt*nx*ny];
                        cursi[kt] = -imsi[kx + ky*nx + kt*nx*ny];
                    }
                }
                
                curAmpW = sqrt(initWr[kx + ky*nx]*initWr[kx + ky*nx] + initWi[kx + ky*nx]*initWi[kx + ky*nx]);
                curAmpF = sqrt(initFr[kx + ky*nx]*initFr[kx + ky*nx] + initFi[kx + ky*nx]*initFi[kx + ky*nx]);

                if(curAmpW>curAmpF) {
                    curPhi = complexPhase(initWr[kx + ky*nx],initWi[kx + ky*nx]);
                }
                else {
                    curPhi = complexPhase(initFr[kx + ky*nx],initFi[kx + ky*nx]);
                }

                curPhi = complexPhase(initWr[kx + ky*nx]+initFr[kx + ky*nx],initWi[kx + ky*nx]+initFi[kx + ky*nx]);

                // Set initial guess.
                Eigen::VectorXd initialGuess(5);
                initialGuess << curAmpW, curAmpF, curPhi, initR2[kx + ky*nx], initFieldmap[kx + ky*nx];

                // Start the optimization.
                auto result = optimizer.minimize(initialGuess);
                
                curAmpW = result.xval(0);
                curAmpF = result.xval(1);
                curPhi = result.xval(2);            

                outWr[kx + ky*nx] = curAmpW*cos(curPhi);
                outWi[kx + ky*nx] = curAmpW*sin(curPhi);
                outFr[kx + ky*nx] = curAmpF*cos(curPhi);
                outFi[kx + ky*nx] = curAmpF*sin(curPhi);
                outR2[kx + ky*nx] = result.xval(3); 
                outFieldmap[kx + ky*nx] = result.xval(4); 
                
            }
            else{
                outWr[kx + ky*nx] = initWr[kx + ky*nx];
                outWi[kx + ky*nx] = initWi[kx + ky*nx];
                outFr[kx + ky*nx] = initFr[kx + ky*nx];
                outFi[kx + ky*nx] = initFi[kx + ky*nx];
                outR2[kx + ky*nx] = initR2[kx + ky*nx]; 
                outFieldmap[kx + ky*nx] = 0; 
            }
     
        }
    }


}
