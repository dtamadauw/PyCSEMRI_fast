#include "fwFit_ComplexLS_1r2star.h"





void fwFit_ComplexLS_1r2star::fitted_line(const Eigen::VectorXd &xval, Eigen::VectorXd &fval)
{
    assert(xval.size() % 2 == 0);

    double shatr, shati, CS, SN, EXP;;
    double curJ1,curJ2,curJ3;
    double curJ4,curJ5,curJ6;
    double expr2, sinfm, cosfm;

    double Wr = xval(0);
    double Wi = xval(1);
    double Fr = xval(2);
    double Fi = xval(3);
    double r2 = xval(4);
    double fieldmap = xval(5);

    // calculate the error vector
    fval.resize(nte*2);
    for(int kt = 0; kt < nte; ++kt)
    {
        CS = cos(2*PI*fieldmap*te[kt]);
        SN = sin(2*PI*fieldmap*te[kt]);
        EXP = exp(-te[kt]*r2);

        shatr = CS*EXP*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) - SN*EXP*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);
        shati = SN*EXP*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) + CS*EXP*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);

        fval(kt) = shatr;
        fval(kt+nte) = shati;
    }
    
}


// Implement an objective functor for Eigen.
struct ParabolicErrorFunctor
{
    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    data_str *data;
    int m_inputs; // Number of parameters (p = 6)
    int m_values; // Number of residuals (n = nte * 2)

    // Constructor to set dimensions
    ParabolicErrorFunctor(data_str *d, int nte) : data(d) 
    {
        m_inputs = 6;       // p = 6 (Wr, Wi, Fr, Fi, R2, FM)
        m_values = nte * 2; // n = 2 * nte
    }

    int operator()(const Eigen::VectorXd &xval, Eigen::VectorXd &fval) const
    {
        // This is your *exact* f(x) calculation from ParabolicError
        int nte = data->nte;
        double *cursr = data->cursr;
        double *cursi = data->cursi;
        double *te = data->te;
        double *swr = data->swr;
        double *swi = data->swi;
        double *sfr = data->sfr;
        double *sfi = data->sfi;

        
        double Wr = xval(0), Wi = xval(1), Fr = xval(2);
        double Fi = xval(3), r2 = xval(4), fieldmap = xval(5);
        double shatr, shati, CS, SN, EXP;

        for(int kt = 0; kt < nte; ++kt)
        {
            CS = cos(2*PI*fieldmap*te[kt]);
            SN = sin(2*PI*fieldmap*te[kt]);
            EXP = exp(-te[kt]*r2);

            shatr = CS*EXP*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) - SN*EXP*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);
            shati = SN*EXP*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) + CS*EXP*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);

            fval(kt) = shatr - cursr[kt];
            fval(kt+nte) = shati - cursi[kt];
        }
        return 0; // Success
    }

    // This computes the Jacobian matrix J
    int df(const Eigen::VectorXd &xval, Eigen::MatrixXd &jacobian) const
    {
        // This is your *exact* Jacobian calculation from ParabolicError
        int nte = data->nte;
        double *cursr = data->cursr;
        double *cursi = data->cursi;
        double *te = data->te;
        double *swr = data->swr;
        double *swi = data->swi;
        double *sfr = data->sfr;
        double *sfi = data->sfi;

        
        double Wr = xval(0), Wi = xval(1), Fr = xval(2);
        double Fi = xval(3), r2 = xval(4), fieldmap = xval(5);
        double shatr, shati, CS, SN, EXP;
        double curJ1, curJ2, curJ3, curJ4, curJ5, curJ6;
        double expr2, sinfm, cosfm;

        for(int kt = 0; kt < nte; ++kt)
        {
            CS = cos(2*PI*fieldmap*te[kt]);
            SN = sin(2*PI*fieldmap*te[kt]);
            EXP = exp(-te[kt]*r2);

            shatr = CS*EXP*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) - SN*EXP*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);
            shati = SN*EXP*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) + CS*EXP*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);
                        
            /*Callback df/di(x)*/
            expr2 = exp(-te[kt]*r2);
            sinfm = sin(2*PI*fieldmap*te[kt]);
            cosfm = cos(2*PI*fieldmap*te[kt]);

            shatr=cosfm*expr2*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) - sinfm*expr2*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);
            shati=sinfm*expr2*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) + cosfm*expr2*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]);

            curJ1 = cosfm*expr2*swr[kt] - sinfm*expr2*swi[kt];
            jacobian(kt,0) = curJ1;
            curJ1 = sinfm*expr2*swr[kt] + cosfm*expr2*swi[kt];
            jacobian(kt+nte,0) = curJ1;

            curJ2 = -cosfm*expr2*swi[kt] - sinfm*expr2*swr[kt];
            jacobian(kt,1) = curJ2;
            curJ2 = -sinfm*expr2*swi[kt] + cosfm*expr2*swr[kt];
            jacobian(kt+nte,1) = curJ2;

            curJ3 = cosfm*expr2*sfr[kt] - sinfm*expr2*sfi[kt];
            jacobian(kt,2) = curJ3;
            curJ3 = sinfm*expr2*sfr[kt] + cosfm*expr2*sfi[kt];
            jacobian(kt+nte,2) = curJ3;

            curJ4 = -cosfm*expr2*sfi[kt] - sinfm*expr2*sfr[kt];
            jacobian(kt,3) = curJ4;
            curJ4 = -sinfm*expr2*sfi[kt] + cosfm*expr2*sfr[kt];
            jacobian(kt+nte,3) = curJ4;

            curJ5 = -te[kt]*shatr;
            jacobian(kt,4) = curJ5;
            curJ5 = -te[kt]*shati;
            jacobian(kt+nte,4) = curJ5;
            
            curJ6 = -2*PI*te[kt]*(sinfm*expr2*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) + cosfm*expr2*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]));
            jacobian(kt,5) = curJ6;
            curJ6 =  2*PI*te[kt]*(cosfm*expr2*(Wr*swr[kt] + Fr*sfr[kt] - Wi*swi[kt] - Fi*sfi[kt]) - sinfm*expr2*(Wr*swi[kt] + Fr*sfi[kt] + Wi*swr[kt] + Fi*sfr[kt]));
            jacobian(kt+nte,5) = curJ6;
        }
        return 0; // Success
    }


    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
    
};



void fwFit_ComplexLS_1r2star::initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in){

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
    fitSr = new double[nx*ny*nte];
    fitSi = new double[nx*ny*nte];
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


void fwFit_ComplexLS_1r2star::fit_all(){

    data_str data;
    data.nte = nte;
    data.cursr = cursr;
    data.cursi = cursi;
    data.te = te;
    data.swr = swr;
    data.swi = swi;
    data.sfr = sfr;
    data.sfi = sfi;

    ParabolicErrorFunctor functor(&data, this->nte);
    Eigen::LevenbergMarquardt<ParabolicErrorFunctor> optimizer(functor);
    optimizer.parameters.maxfev = 50; // Max iterations (function evaluations)
    optimizer.parameters.xtol = 1e-4; // Step tolerance
    optimizer.parameters.gtol = 1e-4; // Gradient tolerance

    double *imsr = imDataParams->images_r;
    double *imsi = imDataParams->images_i;

    // Pre-allocate the other vectors from your old code
    Eigen::VectorXd initialGuess(6);

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
                
                // Set initial guess.
                //Eigen::VectorXd initialGuess(6);
                //initialGuess << initWr[kx + ky*nx], initWi[kx + ky*nx], initFr[kx + ky*nx], initFi[kx + ky*nx], initR2[kx + ky*nx], initFieldmap[kx + ky*nx];
                initialGuess(0) = initWr[kx + ky * nx];
                initialGuess(1) = initWi[kx + ky * nx];
                initialGuess(2) = initFr[kx + ky * nx];
                initialGuess(3) = initFi[kx + ky * nx];
                initialGuess(4) = initR2[kx + ky * nx];
                initialGuess(5) = initFieldmap[kx + ky * nx];

                
                // Start the optimization.
                optimizer.minimize(initialGuess);

                Eigen::VectorXd fitted_vector(nte*2);
                fitted_line(initialGuess, fitted_vector);
                
                for(int kt=0;kt<nte;kt++) {
                    fitSr[kx + ky*nx + kt*nx*ny] = fitted_vector[kt];
                    fitSi[kx + ky*nx + kt*nx*ny] = fitted_vector[kt+nte];
                }

                outWr[kx + ky*nx] = initialGuess(0); 
                outWi[kx + ky*nx] = initialGuess(1);
                outFr[kx + ky*nx] = initialGuess(2); 
                outFi[kx + ky*nx] = initialGuess(3);
                outR2[kx + ky*nx] = initialGuess(4); 
                outFieldmap[kx + ky*nx] = initialGuess(5);
                
                        
            }
            else{
                for(int kt=0;kt<nte;kt++) {
                    fitSr[kx + ky*nx + kt*nx*ny] = 0.0;
                    fitSi[kx + ky*nx + kt*nx*ny] = 0.0;
                }
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
