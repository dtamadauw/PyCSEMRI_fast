#include "fwFit_MagnLS_1r2star.h"



// Implement an objective functor for Eigen.
struct ParabolicError_Magn
{
    constexpr static bool ComputesJacobian = true;
    data_str *data;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval,
                    Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
    {
        assert(xval.size() % 2 == 0);


        double shatr, shati, CS, SN, EXP;;
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


        double Wr = xval(0);
        double Wi = xval(1);
        double Fr = xval(2);
        double Fi = xval(3);
        double r2 = xval(4);
        double fieldmap = xval(5);

        //printf("shat:");

        // calculate the error vector
        fval.resize(nte);
        for(lsqcpp::Index kt = 0; kt < nte; ++kt)
        {


            EXP = exp(-te[kt]*r2);

            shatr = EXP*sqrt((Wr*swr[kt] + Fr*sfr[kt])*(Wr*swr[kt] + Fr*sfr[kt]) + (Wr*swi[kt] + Fr*sfi[kt])*(Wr*swi[kt] + Fr*sfi[kt]));

            /*Callback f(x)*/
            fval(kt) = shatr - cursr[kt];
            //printf("%f, ",shatr);
//
        }
        //printf("\n");
        //printf("fval: %f, %f, %f, %f, %f, %f, \n", cursr[0],cursr[1],cursr[2],cursr[3],cursr[4],cursr[5]);


        // calculate the jacobian explicitly
        jacobian.setZero(fval.size(), xval.size());
        for(lsqcpp::Index kt = 0; kt < nte; ++kt)
        {
            EXP = exp(-te[kt]*r2);

            shatr = sqrt((Wr*swr[kt] + Fr*sfr[kt])*(Wr*swr[kt] + Fr*sfr[kt]) + (Wr*swi[kt] + Fr*sfi[kt])*(Wr*swi[kt] + Fr*sfi[kt])) + 1e-12;
            curJ1 = EXP*(Wr*swr[kt] + Fr*sfr[kt])/shatr;
            jacobian(kt,0) = curJ1;
            curJ2 = EXP*(Fr*(sfr[kt]*sfr[kt] + sfi[kt]*sfi[kt]) + Wr*sfr[kt])/shatr;
            jacobian(kt,2) = curJ2;
            curJ3 = EXP*(-Wr*Wr*te[kt] - Wr*Fr*sfr[kt]*te[kt] -Fr*Fr*te[kt]*(sfr[kt]*sfr[kt] + sfi[kt]*sfi[kt]) - Wr*Fr*sfr[kt]*te[kt])/shatr;
            jacobian(kt,4) = curJ3;
            
        }   
        
    }
};



void fwFit_MagnLS_1r2star::initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in){

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


void fwFit_MagnLS_1r2star::fit_all(){

    printf("In fit_all()\n");

    lsqcpp::LevenbergMarquardtX<double, ParabolicError_Magn> optimizer;
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



    printf("Assign data structure\n");

    data_str data;
    data.nte = nte;
    data.cursr = cursr;
    data.cursi = cursi;
    data.te = te;
    data.swr = swr;
    data.swi = swi;
    data.sfr = sfr;
    data.sfi = sfi;

    ParabolicError_Magn costFunction;
    costFunction.data = &data;
    optimizer.setObjective(costFunction);

    double *imsr = imDataParams->images_r;
    double *imsi = imDataParams->images_i;

    /* Loop over all pixels */
    //int kx = 83;
    //int ky = 60;
    for(int kx=0;kx<nx;kx++) {
        for(int ky=0;ky<ny;ky++) {
        
            if(masksignal[kx + ky*nx] > 0.1){
                /* Get signal at current voxel */
                if(clockwise>0) {
                    for(int kt=0;kt<nte;kt++) {
                        double re = imsr[kx + ky*nx + kt*nx*ny];
                        double im = imsi[kx + ky*nx + kt*nx*ny];
                        cursr[kt] = sqrt(re*re+im*im);
                        cursi[kt] = 0.0;
                    }
                } else {
                    for(int kt=0;kt<nte;kt++) {
                        double re = imsr[kx + ky*nx + kt*nx*ny];
                        double im = imsi[kx + ky*nx + kt*nx*ny];
                        cursr[kt] = sqrt(re*re+im*im);
                        cursi[kt] = 0.0;
                    }
                }
                // Set initial guess.
                Eigen::VectorXd initialGuess(6);
                initialGuess << initWr[kx + ky*nx], initWi[kx + ky*nx], initFr[kx + ky*nx], initFi[kx + ky*nx], initR2[kx + ky*nx], initFieldmap[kx + ky*nx];

                // Start the optimization.
                auto result = optimizer.minimize(initialGuess);
                //printf("(%d,%d): %f, %f=>(%f,%f)\n", kx,ky,initWr[kx + ky*nx],initFr[kx + ky*nx],result.xval(0),result.xval(2));


                outWr[kx + ky*nx] = result.xval(0); 
                outWi[kx + ky*nx] = result.xval(1);
                outFr[kx + ky*nx] = result.xval(2); 
                outFi[kx + ky*nx] = result.xval(3);
                outR2[kx + ky*nx] = result.xval(4); 
                outFieldmap[kx + ky*nx] = result.xval(5); 
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
