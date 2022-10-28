package com.fawai.asr;

import android.Manifest;
import android.annotation.SuppressLint;
import android.net.Uri;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Process;
import android.provider.ContactsContract;
import android.util.Log;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.CheckBox;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import ai.onnxruntime.OrtException;

public class MainActivity extends AppCompatActivity {

  private final int MY_PERMISSIONS_RECORD_AUDIO = 1;
  private final int MY_PERMISSIONS_READ_CONTACT = 2;
  private static final String LOG_TAG = "FAWASR";
  private static final int SAMPLE_RATE = 16000;  // The sampling rate
  private static final int MAX_QUEUE_SIZE = 2500;  // 100 seconds audio, 1 / 0.04 * 100
  private static final List<String> resource = Arrays.asList(
          "final.zip", "units.txt", "ctc.ort", "decoder.ort", "encoder.ort", "context.txt"
  );

  private boolean startRecord = false;
  private AudioRecord record = null;
  private int miniBufferSize = 0;  // 1280 bytes 648 byte 40ms, 0.04s
  private final BlockingQueue<short[]> asrBufferQueue = new ArrayBlockingQueue<>(MAX_QUEUE_SIZE);
  private final BlockingQueue<short[]> vadBufferQueue = new ArrayBlockingQueue<>(MAX_QUEUE_SIZE);

  private boolean vadBufferClean = false;
  private final BlockingQueue<short[]> vadPreBufferQueue = new ArrayBlockingQueue<>(640 * 3);

  private boolean voiceDetected = false;

  public static void assetsInit(Context context) throws IOException {
    AssetManager assetMgr = context.getAssets();
    // Unzip all files in resource from assets to context.
    // Note: Uninstall the APP will remove the resource files in the context.
    for (String file : assetMgr.list("")) {
      if (resource.contains(file)) {
        File dst = new File(context.getFilesDir(), file);
        if (!dst.exists() || dst.length() == 0) {
          Log.i(LOG_TAG, "Unzipping " + file + " to " + dst.getAbsolutePath());
          InputStream is = assetMgr.open(file);
          OutputStream os = new FileOutputStream(dst);
          byte[] buffer = new byte[4 * 1024];
          int read;
          while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
          }
          os.flush();
        }
      }
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode,
      String[] permissions, int[] grantResults) {
    if (requestCode == MY_PERMISSIONS_RECORD_AUDIO) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Log.i(LOG_TAG, "record permission is granted");
        initRecorder();
      } else {
        Toast.makeText(this, "Permissions denied to record audio", Toast.LENGTH_LONG).show();
        Button button = findViewById(R.id.button);
        button.setEnabled(false);
      }
    }
  }

  @SuppressLint("SetTextI18n")
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);  // formal
    setContentView(R.layout.activity_main);  // formal, R is a class which content the resource ID

    requestAudioPermissions();
    requestContactPermissions();

    try {
      assetsInit(this);
      VoiceDetector.init(getApplicationContext());
    } catch (IOException | OrtException e) {
      Log.e(LOG_TAG, "Error process asset files to file path");
    }

    TextView textView = findViewById(R.id.textView);  // get textView controller
    textView.setText("");  // clear textView

    CheckBox hotWordCheckBox = findViewById(R.id.hotWordCheckBox);  // get hotWordCheckBox controller

    Recognize.init(getFilesDir().getPath(), false);

    final boolean[] updateRecognize = {false};

    hotWordCheckBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
        updateRecognize[0] = true;
      }
    });
    Button button = findViewById(R.id.button);  // get button controller
    button.setText("Start Record");  // set button text
    button.setOnClickListener(view -> {  // watch if button is touched
      if (updateRecognize[0]) {
        if (hotWordCheckBox.isChecked()) {
          Recognize.init(getFilesDir().getPath(), true);
        } else {
          Recognize.init(getFilesDir().getPath(), false);
        }
        updateRecognize[0] = false;
      }

      if (!startRecord) {
        startRecord = true;  // set recording flag
        Recognize.reset();  // reset ASR engine
        startRecordThread();  // start recorder
        startAsrThread();  // start the engine
        startVadThread();
        Recognize.startDecode();  // start ASR decoding
        button.setText("Stop Record");  // set button text
      } else {
        vadBufferClean = false;
        startRecord = false;  // set recording flag
        Recognize.setInputFinished();  // stop ASR engine
        button.setText("Start Record");  // set button text
      }
      button.setEnabled(false);
    });
  }

  private void requestAudioPermissions() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this,
          new String[]{Manifest.permission.RECORD_AUDIO},
          MY_PERMISSIONS_RECORD_AUDIO);
    } else {
      initRecorder();
    }
  }

  private void requestContactPermissions() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this,
              new String[]{Manifest.permission.READ_CONTACTS},
              MY_PERMISSIONS_READ_CONTACT);
    }
  }

  private void initRecorder() {
    // buffer size in bytes 1280
    miniBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT);
    if (miniBufferSize == AudioRecord.ERROR || miniBufferSize == AudioRecord.ERROR_BAD_VALUE) {
      Log.e(LOG_TAG, "Audio buffer can't initialize!");
      return;
    }
    record = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        miniBufferSize);
    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }
    Log.i(LOG_TAG, "Record init okay");
  }

  private void startRecordThread() {
    new Thread(() -> {
      VoiceRectView voiceView = findViewById(R.id.voiceRectView);
      record.startRecording();
      Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
      while (startRecord) {
        short[] buffer = new short[miniBufferSize / 2]; // 640 samples
        int read = record.read(buffer, 0, buffer.length);
        voiceView.add(calculateDb(buffer));
        try {
          if (AudioRecord.ERROR_INVALID_OPERATION != read) {
            if (voiceDetected) {
              if (vadPreBufferQueue.size() > 0) {
                asrBufferQueue.put(vadPreBufferQueue.take()); // put last buffer before voiceDetected
                asrBufferQueue.put(vadBufferQueue.take()); // put buffer in vad
                vadBufferClean = true;
              }

              asrBufferQueue.put(buffer);
            } else {
              vadBufferQueue.put(buffer);
            }
          }
        } catch (InterruptedException e) {
          Log.e(LOG_TAG, e.getMessage());
        }
        Button button = findViewById(R.id.button);
        if (!button.isEnabled() && startRecord) {
          runOnUiThread(() -> button.setEnabled(true));
        }
      }
      record.stop();
      voiceView.zero();
    }).start();
  }

  private void startVadThread() {
    new Thread(() -> {
      while (startRecord || vadBufferQueue.size() > 0 || !voiceDetected) {
        try {
          short[] data = vadBufferQueue.take();
          voiceDetected = VoiceDetector.vadDetect(data);
          if (voiceDetected) {
            vadPreBufferQueue.put(data);
            runOnUiThread(() -> {
              TextView textView = findViewById(R.id.textView);
              textView.setText("VoiceDetected");
            });
            break;
          }
        } catch (InterruptedException | OrtException e) {
          Log.e(LOG_TAG, e.getMessage());
        }
        if (!startRecord) {
          if (!voiceDetected) {
            // enable button if voice not detected
            runOnUiThread(() -> {
              Button button = findViewById(R.id.button);
              button.setEnabled(true);
            });
          }
          break;
        }
      }
    }).start();
  }

  private void startAsrThread() {
    new Thread(() -> {
      // Send all data
      while (startRecord || vadBufferClean || asrBufferQueue.size() > 0) {
        try {
          short[] data = asrBufferQueue.take();
          // 1. add data to C++ interface
          Recognize.acceptWaveform(data);
          // 2. get partial result
          runOnUiThread(() -> {
            TextView textView = findViewById(R.id.textView);
            textView.setText(Recognize.getResult());
          });
        } catch (InterruptedException e) {
          Log.e(LOG_TAG, e.getMessage());
        }
      }

      // Wait for final result
      while (true) {
        // get result
        if (!Recognize.getFinished()) {
          runOnUiThread(() -> {
            TextView textView = findViewById(R.id.textView);
            textView.setText(Recognize.getResult());
          });
        } else {
          runOnUiThread(() -> {
            Button button = findViewById(R.id.button);
            button.setEnabled(true);
          });
          String asrResult = Recognize.getResult();
          boolean callPhoneStatus = asrResult.contains("打电话");
          if (callPhoneStatus) {
            TextView textView = findViewById(R.id.textView);
            int contactBE = asrResult.indexOf("@");
            int contactED = asrResult.lastIndexOf("@");

            if (contactBE == -1 | contactED == -1) {
              Log.i(LOG_TAG, "Not contact intent ");
              textView.setText("未匹配到联系人实体");
            } else {
              String contactName = asrResult.substring(contactBE+1, contactED);
              Log.i(LOG_TAG, "Contact name: " + contactName);
              String number = getContact(contactName);
              if (!number.equals("")) {
                Intent intent = new Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + number));
                startActivity(intent);
              } else {
                Log.i(LOG_TAG, "Not contact name " + contactName);
                textView.setText("未找到所述联系人");
              }
            }
          }
          break;
        }
      }
      voiceDetected = false;

    }).start();
  }

  String getContact(String nameStr) {
    String[] SQL_COLUMN = new String[]{
            ContactsContract.CommonDataKinds.Identity.RAW_CONTACT_ID,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER};
    String mSelectClaus = ContactsContract.Contacts.DISPLAY_NAME+"=?";
    String[] mSelectionArgs = new String[]{nameStr};
    Log.i(LOG_TAG, "Start query contact");
    Cursor cursor = getContentResolver().query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI, SQL_COLUMN, mSelectClaus, mSelectionArgs, null);
    Log.i(LOG_TAG, "Finished query contact");
    if (cursor != null) {
      String ID = "";
      String contactName = "";
      String phoneNumber = "";

      while (cursor.moveToNext()) {
        ID = cursor.getString(0);
        contactName = cursor.getString(1);
        phoneNumber = cursor.getString(2);
        Log.i(LOG_TAG, "Contact name: " + contactName + " Phone number: " + phoneNumber);
      }
      return phoneNumber;
    } else {
      return "";
    }
  }

  private double calculateDb(short[] buffer) {
    double energy = 0.0;
    for (short value : buffer) {
      energy += value * value;
    }
    energy /= buffer.length;
    energy = (10 * Math.log10(1 + energy)) / 100;
    energy = Math.min(energy, 1.0);
    return energy;
  }
}