package com.example.root.harapp;

import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;

// Sensor Imports
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;


//Text to Speech
import android.speech.tts.TextToSpeech;
import android.widget.EditText;


// Java Imports
import java.math.BigDecimal;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.lang.Math;


public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener{

    private static final int N_SAMPLES = 128;

    //Accelerometer Lists
    private static List<Float> accX;
    private static List<Float> accY;
    private static List<Float> accZ;

    //Gyroscope Lists
    private static List<Float> gyroX;
    private static List<Float> gyroY;
    private static List<Float> gyroZ;

    //TextViews
    private EditText laying;
    private EditText sitting;
    private EditText standing;
    private EditText upstairs;
    private EditText walking;
    private EditText downstairs;
    private EditText running;



    //Accelerometer and Gyroscope TextViews
    private EditText accelerometer;
    private EditText gyroscope;

    //TTS Variables
    private TextToSpeech textToSpeech;


    // Time Variables
    private Long lastAccTimer = 0L;
    private Long lastGyroTimer = 0L;
    private Long startTime = 0L;
    private EditText reading;
    private EditText time;


    private HARClassifier classifier;

    //Sensor Manager
    private SensorManager mSensorManager;
    private Sensor accSensor, gyroSensor;

    private String[] labels = {"Downstairs", "Upstairs", "Walking", "Sitting", "Laying", "Standing"};
    private float[] predictions;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        accX = new ArrayList<>();
        accY = new ArrayList<>();
        accZ = new ArrayList<>();

        gyroX = new ArrayList<>();
        gyroY = new ArrayList<>();
        gyroZ = new ArrayList<>();

        downstairs = findViewById(R.id.downstairs);
        upstairs = findViewById(R.id.upstairs);
        laying =  findViewById(R.id.laying);
        sitting = findViewById(R.id.sitting);
        walking = findViewById(R.id.walking);
        running = findViewById(R.id.running);
        standing = findViewById(R.id.standing);

        time = findViewById(R.id.time);
        reading = findViewById(R.id.reading);
        gyroscope =  findViewById(R.id.gyroscope);
        accelerometer =  findViewById(R.id.accelerometer);


        classifier = new HARClassifier(getApplicationContext());

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);

        accSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        gyroSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this,gyroSensor, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this,accSensor, SensorManager.SENSOR_DELAY_GAME);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        long currentTime = (new Date()).getTime() + (event.timestamp - System.nanoTime()) / 1000000L;
        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION){

            if(lastAccTimer == 0 && accX.size() < N_SAMPLES){
                lastAccTimer =  currentTime;
                startTime = currentTime;
                accX.add(event.values[0]);
                accY.add(event.values[1]);
                accZ.add(event.values[2]);
                accelerometer.setText(Float.toString(roundOff(event.values[0]))+","+Float.toString(roundOff(event.values[1]))+","+Float.toString(roundOff(event.values[2])));

            }else{

                long timeDifference = currentTime - lastAccTimer;
                if(timeDifference >= 20 && accX.size() < N_SAMPLES){
                    lastAccTimer =  currentTime;
                    accX.add(event.values[0]);
                    accY.add(event.values[1]);
                    accZ.add(event.values[2]);
                    accelerometer.setText(Float.toString(roundOff(event.values[0]))+","+Float.toString(roundOff(event.values[1]))+","+Float.toString(roundOff(event.values[2])));
                }
            }


        }
        if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE){

            if(lastGyroTimer == 0 && gyroX.size() < N_SAMPLES){
                lastGyroTimer =  currentTime;
                startTime = currentTime;
                gyroX.add(event.values[0]);
                gyroY.add(event.values[1]);
                gyroZ.add(event.values[2]);
                gyroscope.setText(Float.toString(roundOff(event.values[0]))+","+Float.toString(roundOff(event.values[1]))+","+Float.toString(roundOff(event.values[2])));

            }else{

                long timeDifference = currentTime - lastGyroTimer;
                if(timeDifference >= 20 && gyroX.size() < N_SAMPLES){
                    lastGyroTimer =  currentTime;
                    gyroX.add(event.values[0]);
                    gyroY.add(event.values[1]);
                    gyroZ.add(event.values[2]);
                    gyroscope.setText(Float.toString(roundOff(event.values[0]))+","+Float.toString(roundOff(event.values[1]))+","+Float.toString(roundOff(event.values[2])));
                }
            }
        }
        predictActivity(currentTime);
    }

    private void predictActivity(long eventTime) {
        if(accX.size() == N_SAMPLES && accY.size() == N_SAMPLES && accZ.size() == N_SAMPLES && gyroX.size() == N_SAMPLES && gyroY.size() == N_SAMPLES && gyroZ.size() == N_SAMPLES){
            List<Float> data = new ArrayList<>();
            data.addAll(accX);
            data.addAll(accY);
            data.addAll(accZ);
            data.addAll(gyroX);
            data.addAll(gyroY);
            data.addAll(gyroZ);
            predictions = classifier.predictProbabilities(toFloatArray(data));

            downstairs.setText(Float.toString(roundOff(predictions[2])));
            laying.setText(Float.toString(roundOff(predictions[5])));
            sitting.setText(Float.toString(roundOff(predictions[3])));
            standing.setText(Float.toString(roundOff(predictions[4])));
            upstairs.setText(Float.toString(roundOff(predictions[1])));
            walking.setText(Float.toString(roundOff(predictions[0])));
            running.setText(Float.toString(roundOff(predictions[0])));

            Date date = new Date(eventTime - startTime);
            DateFormat formatter = new SimpleDateFormat("HH:mm:ss:SSS");
            String dateFormatted = formatter.format(date);

            reading.setText(Integer.toString(accX.size()));
            time.setText(dateFormatted);

            clearData();



        }

    }


    private void clearData(){
        accX.clear();
        accY.clear();
        accZ.clear();
        gyroX.clear();
        gyroY.clear();
        gyroZ.clear();
        startTime = 0L;
        lastGyroTimer = 0L;
        lastAccTimer = 0L;


    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (predictions == null || predictions.length == 0) {
                    return;
                }
                float max = -1;
                int idx = -1;
                for (int i = 0; i < predictions.length; i++) {
                    if (predictions[i] > max) {
                        idx = i;
                        max = predictions[i];
                    }
                }
                textToSpeech.speak(labels[idx], TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));

            }
        }, 2000, 5000);
    }



    protected void onPause() {
        mSensorManager.unregisterListener(this);
        super.onPause();
    }

    protected void onResume(){
        super.onResume();
        mSensorManager.registerListener(this,gyroSensor, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this,accSensor, SensorManager.SENSOR_DELAY_GAME);
    }


    private static float roundOff(float d){
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(2,BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();


    }

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }
}
