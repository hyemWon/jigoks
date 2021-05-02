package com.example.jigoks;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.animation.AnimatorInflater;
import android.animation.AnimatorSet;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.Locale;

import static android.speech.tts.TextToSpeech.ERROR;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, TextToSpeech.OnInitListener {
    ImageView micImage;
    TextView number;
    TextView info;
    String value = null;

    private TextToSpeech ttsClient;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        micImage = (ImageView)findViewById(R.id.mic);
        number = (TextView)findViewById(R.id.bus_number);
        info = (TextView)findViewById(R.id.info);

        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference ref = database.getReference("bus");

        ref.setValue("버스 안내를 시작합니다.");


        // 데이터베이스의 내용이 변동되면 콜백함수를 계속 호출
        ref.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                value = snapshot.getValue(String.class);
                // 데이터를 화면에 출력해 준다.
                Log.d("BUS", "Value is: " + value);
                if (value == "버스 안내를 시작합니다."){
                    info.setText(value);
                    onInit(1);
                }else{
                    number.setText(value+"번");
                    info.setText("버스가 들어오고 있습니다.");
                    ttsClient.speak(value + "번 버스가 들어오고 있습니다.", TextToSpeech.QUEUE_FLUSH, null);
                }
                animation();
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                Log.w("BUS", "Failed to read value.", error.toException());
            }
        });
        ttsClient = new TextToSpeech(getApplicationContext(), this);

        ((ImageView)findViewById(R.id.mic)).setOnClickListener(this);
    }

    public void animation(){
        final Animation hyperspaceJump = AnimationUtils.loadAnimation(MainActivity.this, R.anim.rotate);
        micImage.startAnimation(hyperspaceJump);
    }

    @Override
    public void onClick(View v) {
        if(v.getId() == R.id.mic){
            if (value == "버스 안내를 시작합니다."){
                onInit(1);
            }else{
                ttsClient.speak(value + "번 버스가 들어오고 있습니다.", TextToSpeech.QUEUE_FLUSH, null);
            }
            animation();
        }
    }

    @Override
    public void onInit(int status) {
        if(status != ERROR) {
            // 언어를 선택한다.
            ttsClient.setLanguage(Locale.KOREAN);
            ttsClient.speak("버스 안내를 시작합니다.", TextToSpeech.QUEUE_FLUSH, null);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // TTS 객체가 남아있다면 실행을 중지하고 메모리에서 제거한다.
        if(ttsClient != null){
            ttsClient.stop();
            ttsClient.shutdown();
            ttsClient = null;
        }
    }


}
