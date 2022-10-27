package com.fawai.asr;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.ejml.simple.SimpleMatrix;

public class AudioFeatureExtraction {
    private static final String LOG_TAG = "AUDIO_FEATURE";

    private static final int n_fft = 512;
    private static final int n_mfcc = 64;
    private static final int hop_length = 160;
    private static final int win_length = 400;
    private final double[][] melBasis;
    private final double[][] dctBasis;

    public AudioFeatureExtraction(double[][] fb, double[][] dct) {melBasis=fb; dctBasis=dct;}

    public float[] generateMFCCFeatures(float[] magValues) {
        return this.dctMfcc(magValues);
    }

    public double[][] extractSTFTFeatures(float[] y) {
        double[] fftwin = this.getWindow();
        double[][] frame = this.padFrame(y);
        double[][] fftmagSpec = new double[1 + n_fft / 2][frame[0].length];
        double[] fftFrame = new double[n_fft];

        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);

        for(int k = 0; k < frame[0].length; ++k) {
            int fftFrameCounter = 0;

            for(int l = 0; l < n_fft; ++l) {
                fftFrame[fftFrameCounter] = fftwin[l] * frame[l][k];
                ++fftFrameCounter;
            }

            try {
                Complex[] complx = transformer.transform(fftFrame, TransformType.FORWARD);

                for(int i = 0; i < 1 + n_fft / 2; ++i) {
                    double rr = complx[i].getReal();
                    double ri = complx[i].getImaginary();

                    fftmagSpec[i][k] = rr * rr + ri * ri;
                }
            } catch (IllegalArgumentException var17) {
                System.out.println(var17);
            }
        }

        return fftmagSpec;
    }

    public double[][] melSpectrogram(float[] y) {
        double[][] spectro = this.extractSTFTFeatures(y);
        double[][] melS = new double[this.melBasis.length][spectro[0].length];

        SimpleMatrix H = new SimpleMatrix(this.melBasis);
        SimpleMatrix P = new SimpleMatrix(spectro);
        SimpleMatrix C = H.mult(P);

        for (int r = 0; r < C.numRows(); r++) {
            for (int c = 0; c < C.numCols(); c++) {
                melS[r][c] = C.get(r, c);
            }
        }
        return melS;
    }

    private double[][] powerToDb(double[][] melS) {
        double[][] log_spec = new double[melS.length][melS[0].length];

        for(int i = 0; i < melS.length; ++i) {
            for(int j = 0; j < melS[0].length; ++j) {
                log_spec[i][j] = Math.log(0.000001D + melS[i][j]);
            }
        }

        return log_spec;
    }

    private float[] dctMfcc(float[] y) {
        double[][] specTroGram = this.powerToDb(this.melSpectrogram(y));
        float[] mfccSpecTro = new float[n_mfcc * specTroGram[0].length];

        SimpleMatrix H = new SimpleMatrix(dctBasis);
        SimpleMatrix P = new SimpleMatrix(specTroGram);
        SimpleMatrix C = H.mult(P);

        for (int r = 0; r < C.numRows(); r++) {
            for (int c = 0; c < C.numCols(); c++) {
                mfccSpecTro[r * C.numCols() + c] = (float) C.get(r, c);
            }
        }

        return mfccSpecTro;
    }

    private double[] getWindow() {
        double[] win = new double[n_fft];
        int pad_len = (int) ((n_fft - win_length) / 2);
        for(int i = pad_len; i < win_length + pad_len; ++i) {
            win[i] = 0.5D - 0.5D * Math.cos(6.283185307179586D * (double)(i-pad_len) / (double) win_length);
        }

        return win;
    }

    private double[][] padFrame(float[] yValues) {
        double[][] frame = (double[][])null;
        double[] ypad;
        int j;

        ypad = new double[n_fft + yValues.length];

        for(j = 0; j < n_fft / 2; ++j) {
            ypad[n_fft / 2 - j - 1] = (double)yValues[j + 1];
            ypad[n_fft / 2 + yValues.length + j] = (double)yValues[yValues.length - 2 - j];
        }

        for(j = 0; j < yValues.length; ++j) {
            ypad[n_fft / 2 + j] = (double)yValues[j];
        }

        frame = this.yFrame(ypad);

        return frame;
    }

    private double[][] yFrame(double[] ypad) {
        int n_frames = 1 + (ypad.length - n_fft) / hop_length;
        double[][] winFrames = new double[n_fft][n_frames];

        for(int i = 0; i < n_fft; ++i) {
            for(int j = 0; j < n_frames; ++j) {
                winFrames[i][j] = ypad[j * hop_length + i];
            }
        }

        return winFrames;
    }
}
