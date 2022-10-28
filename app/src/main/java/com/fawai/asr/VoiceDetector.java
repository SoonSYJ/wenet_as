package com.fawai.asr;

import android.content.Context;
import android.util.Log;

import ai.onnxruntime.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Scanner;


public class VoiceDetector {
    private static final String LOG_TAG = "FAWASR";

    private static String tVadModelFile = "MarbleNet-mfa.ort";
    private static OrtEnvironment ortVadEnvironment;
    private static OrtSession ortVadSession;

    private static AudioFeatureExtraction featureEngine = null;

    private final static int CHUNK_TO_READ = 10;
    private final static int CHUNK_SIZE = 640;
    public final static int INPUT_SIZE = CHUNK_SIZE * CHUNK_TO_READ;
    private final static int FEAT_FRAME_SIZE = INPUT_SIZE / 160 + 1;
    private final static int FEAT_DIM = 64;

    private final static float[] floatInputBuffer = new float[INPUT_SIZE];
    private static boolean bufferReady = false;
    private static int bufferSize = 0;

    protected static void init(Context context) throws IOException, OrtException {
        // load feature extractor model
//        mModuleFeature = Module.load(assetFilePath(context, tFeatModelFile));

        double[][] melBasis = new double[64][257];
        int lineCount = 0;
        try (Scanner sc = new Scanner(new FileReader(assetFilePath(context, "fb_m.txt")))) {
            while (sc.hasNextLine()) {  //按行读取字符串
                String line = sc.nextLine();
                String[] ss = line.split(",");
                for (int i = 0; i < 257; i++) {melBasis[lineCount][i] = Double.parseDouble(ss[i]);}
                lineCount += 1;
            }
        }

        double[][] dctBasis = new double[64][64];
        lineCount = 0;
        try (Scanner sc = new Scanner(new FileReader(assetFilePath(context, "dct_m.txt")))) {
            while (sc.hasNextLine()) {  //按行读取字符串
                String line = sc.nextLine();
                String[] ss = line.split(",");
                for (int i = 0; i < 64; i++) {dctBasis[lineCount][i] = Double.parseDouble(ss[i]);}
                lineCount += 1;
            }
        }

        featureEngine = new AudioFeatureExtraction(melBasis, dctBasis);

        // load vad model
        InputStream modelIn = context.getAssets().open(tVadModelFile);
        int length = modelIn.available();
        byte[] buffer = new byte[length];
        modelIn.read(buffer);

        ortVadEnvironment = OrtEnvironment.getEnvironment();
        ortVadSession = ortVadEnvironment.createSession(buffer);
        Log.e(LOG_TAG, "Vad ort env init success");
    }

    private static void bufferBucket(short[] inputBuffer) {
        for (int i = 0; i < inputBuffer.length; ++i) {
            floatInputBuffer[bufferSize + i] = inputBuffer[i] / (float)Short.MAX_VALUE;  // from short to float
        }
        bufferSize = bufferSize + inputBuffer.length;
        if (bufferSize == INPUT_SIZE) {
            bufferReady = true;
        }
    }

    protected static boolean vadDetect(short[] inputBuffer) throws OrtException {
        bufferBucket(inputBuffer);
        if (!bufferReady) {
            return false;
        }
        // clear buffer bucket
        bufferReady = false;
        bufferSize = 0;

        float[] mfccFeatures = featureEngine.generateMFCCFeatures(floatInputBuffer);

        // vad model
        String inputName = ortVadSession.getInputNames().iterator().next();
        OnnxTensor input = OnnxTensor.createTensor(
                ortVadEnvironment, FloatBuffer.wrap(mfccFeatures),
                new long[]{1, FEAT_DIM, FEAT_FRAME_SIZE});
        OrtSession.Result output = ortVadSession.run(Collections.singletonMap(inputName, input));

        // voice detection
        // TODO sos and eos parsing
        float[][][] logprob = (float[][][]) output.get(0).getValue();
        for (int i = 0; i < logprob[0].length; i++) {
            Log.d(LOG_TAG, "vad log prob: " + logprob[0][i][0]);
            if (logprob[0][i][0] < 0.5)
                return true;
        }
        return false;
    }

    private static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(LOG_TAG, assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }
}
