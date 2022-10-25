package com.fawai.asr;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import ai.onnxruntime.*;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

public class VoiceDetector {
    private static final String LOG_TAG = "FAWASR";

    private static String tVadModelFile = "MarbleNet-mfa.ort";
    private static OrtEnvironment ortVadEnvironment;
    private static OrtSession ortVadSession;

    private final static String tFeatModelFile = "mfcc_feat_infer.ptl";
    private static Module mModuleFeature;

    private final static int CHUNK_TO_READ = 10;
    private final static int CHUNK_SIZE = 640;
    public final static int INPUT_SIZE = CHUNK_SIZE * CHUNK_TO_READ;
    private final static int FEAT_FRAME_SIZE = INPUT_SIZE / 160 + 1;
    private final static int FEAT_DIM = 64;

    protected static void init(Context context) throws IOException, OrtException {
        // load feature extractor model
        mModuleFeature = Module.load(assetFilePath(context, tFeatModelFile));

        // load vad model
        InputStream modelIn = context.getAssets().open(tVadModelFile);
        int length = modelIn.available();
        byte[] buffer = new byte[length];
        modelIn.read(buffer);

        ortVadEnvironment = OrtEnvironment.getEnvironment();
        ortVadSession = ortVadEnvironment.createSession(buffer);
        Log.e(LOG_TAG, "Vad ort env init success");
    }

    private int vadDetect(short[] inputBuffer) throws OrtException {
        double[] floatInputBuffer = new double[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            floatInputBuffer[i] = inputBuffer[i] / (float)Short.MAX_VALUE;  // from short to float
        }

        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(INPUT_SIZE);
        for (int i = 0; i < floatInputBuffer.length - 1; i++) {
            inTensorBuffer.put((float) floatInputBuffer[i]);
        }

        final Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, INPUT_SIZE});

        // model forward
        final long startTime = SystemClock.elapsedRealtime();
        // feature
        IValue feat = mModuleFeature.forward(IValue.from(inTensor));
        final long featTime = SystemClock.elapsedRealtime() - startTime;
        Tensor feat_t = feat.toTensor();
        // vad model
        String inputName = ortVadSession.getInputNames().iterator().next();
        OnnxTensor input = OnnxTensor.createTensor(
                ortVadEnvironment, FloatBuffer.wrap(feat_t.getDataAsFloatArray()),
                new long[]{1, FEAT_DIM, FEAT_FRAME_SIZE});
        final long featTransTime = SystemClock.elapsedRealtime() - startTime - featTime;
        OrtSession.Result output = ortVadSession.run(Collections.singletonMap(inputName, input));

        // voice detection
        // TODO sos and eos parsing
        float[][][] logprob = (float[][][]) output.get(0).getValue();
        for (int i = 0; i < logprob.length; i++) {
            Log.d(LOG_TAG, "vad log prob: " + logprob[0][0][i]);
            if (logprob[0][0][i] < 0.5)
                return 1;
        }
        return 0;
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
