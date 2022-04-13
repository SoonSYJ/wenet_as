package com.fawai.asr;

import android.Manifest;
import android.net.Uri;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class MainActivity extends AppCompatActivity {

  private final int MY_PERMISSIONS_RECORD_AUDIO = 1;
  private final int MY_PERMISSIONS_READ_CONTACT = 2;
  private static final String LOG_TAG = "WENET";
  private static final int SAMPLE_RATE = 16000;  // The sampling rate
  private static final int MAX_QUEUE_SIZE = 2500;  // 100 seconds audio, 1 / 0.04 * 100

  private boolean startRecord = false;
  private AudioRecord record = null;
  private int miniBufferSize = 0;  // 1280 bytes 648 byte 40ms, 0.04s
  private final BlockingQueue<short[]> bufferQueue = new ArrayBlockingQueue<>(MAX_QUEUE_SIZE);

  public static String assetFilePath(Context context, String assetName) {
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
      Log.e(LOG_TAG, "Error process asset " + assetName + " to file path");
    }
    return null;
  }

  @Override
  public void onRequestPermissionsResult(int requestCode,
      String[] permissions, int[] grantResults) {
    if (requestCode == MY_PERMISSIONS_RECORD_AUDIO) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Log.i(LOG_TAG, "record permission is granted");
        initRecoder();
      } else {
        Toast.makeText(this, "Permissions denied to record audio", Toast.LENGTH_LONG).show();
        Button button = findViewById(R.id.button);
        button.setEnabled(false);
      }
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);  // formal
    setContentView(R.layout.activity_main);  // formal, R is a class which content the resource ID

    requestAudioPermissions();
    requestContactPermissions();

    final String modelPath = new File(assetFilePath(this, "final.zip")).getAbsolutePath();
    final String dictPath = new File(assetFilePath(this, "words.txt")).getAbsolutePath();
    final String contextPath = new File(assetFilePath(this, "context.txt")).getAbsolutePath();
    TextView textView = findViewById(R.id.textView);  // get textView controller
    textView.setText("");  // clear textView

    CheckBox hotWordCheckBox = findViewById(R.id.hotWordCheckBox);  // get hotWordCheckBox controller

    Recognize.init(modelPath, dictPath, "");
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
          Recognize.init(modelPath, dictPath, contextPath);
        } else {
          Recognize.init(modelPath, dictPath, "");
        }
        updateRecognize[0] = false;
      }

      if (!startRecord) {
        startRecord = true;  // set recording flag
        Recognize.reset();  // reset ASR engine
        startRecordThread();  // start recorder
        startAsrThread();  // start the engine
        Recognize.startDecode();  // start ASR decoding
        button.setText("Stop Record");  // set button text
      } else {
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
      initRecoder();
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

  private void initRecoder() {
    // buffer size in bytes 1280
    miniBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT);
    if (miniBufferSize == AudioRecord.ERROR || miniBufferSize == AudioRecord.ERROR_BAD_VALUE) {
      Log.e(LOG_TAG, "Audio buffer can't initialize!");
      return;
    }
    record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT,
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
        short[] buffer = new short[miniBufferSize / 2];
        int read = record.read(buffer, 0, buffer.length);
        voiceView.add(calculateDb(buffer));
        try {
          if (AudioRecord.ERROR_INVALID_OPERATION != read) {
            bufferQueue.put(buffer);
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

  private void startAsrThread() {
    new Thread(() -> {
      // Send all data
      while (startRecord || bufferQueue.size() > 0) {
        try {
          short[] data = bufferQueue.take();
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
            int contactBE = asrResult.indexOf("<c>");
            int contactED = asrResult.lastIndexOf("</c>");

            if (contactBE == -1 | contactED == -1) {
              Log.i(LOG_TAG, "Not contact intent ");
              textView.setText("^^未匹配到联系人实体!");
            } else {
              String contactName = asrResult.substring(contactBE + 3, contactED);
              Log.i(LOG_TAG, "Contact name: " + contactName);
              String number = getContact(contactName);
              if (number != "") {
                Intent intent = new Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + number));
                startActivity(intent);
              } else {
                Log.i(LOG_TAG, "Not contact name " + contactName);
                textView.setText("^^未找到所述联系人！");
              }
            }
          }
          break;
        }
      }
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
}