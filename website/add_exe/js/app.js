const firebaseConfig = {
    apiKey: "AIzaSyCsT5QrQpzyQTyaB3Dq7qIDrNMJEoODPpI",
    authDomain: "alpha-f18cd.firebaseapp.com",
    databaseURL: "https://alpha-f18cd-default-rtdb.firebaseio.com",
    projectId: "alpha-f18cd",
    storageBucket: "alpha-f18cd.appspot.com",
    messagingSenderId: "532001314322",
    appId: "1:532001314322:web:a6353cee49e5fa4f7caacb",
    measurementId: "G-SHW9Z6D14W"
    };
// Initialize Firebase
firebase.initializeApp(firebaseConfig);

//firebase를 전역 변수로 설정
console.log(firebase)

//인증 서비스 제공 업체
var provider = new firebase.auth.GoogleAuthProvider();

//사용자 인증
const auth = firebase.auth();

const whenSignedIn = document.getElementById('whenSignedIn');
const whenSignedOut = document.getElementById('whenSignedOut');

const btnGoogle = document.getElementById('btnGoogle');
const signOutBtn = document.getElementById('signOutBtn');
const signOutBtn_li = document.getElementById('signOutBtn_li');

const userDetails = document.getElementById('userDetails');

const container1 = document.getElementById('container1');


const userLoginEmail = document.getElementById('userLoginEmail');
const userLoginPassword = document.getElementById('userLoginPassword');
const btnLogin = document.getElementById('btnLogin');
const btnSignUp = document.getElementById('btnSignUp');





// Add login event
btnLogin.addEventListener('click', e => {
    // Get email and pass
    const email = userLoginEmail.value;
    const pass = userLoginPassword.value;
    const auth = firebase.auth();

    // Sign in
    const promise = auth.signInWithEmailAndPassword(email,pass);
    promise.catch(e => {
        console.log(e.message)
        alert("PASSWORD incorrect")
    });
});


// Add Enter Key login
userLoginPassword.addEventListener("keyup", function(event) {
// Number 13 is the "Enter" key on the keyboard
if (event.keyCode === 13) {
    // Cancel the default action, if needed
    // event.preventDefault();
    // Trigger the button element with a click
    btnLogin.click();
}
});





// Add signup event
btnSignUp.addEventListener('click', e => {
    // Get email and pass
    // TODO: CHECK 4 REAL EMAILZ
    const email = userLoginEmail.value;
    const pass = userLoginPassword.value;
    const auth = firebase.auth();

    // Sign in
    const promise = auth.createUserWithEmailAndPassword(email,pass);
    promise.catch(e => {
        console.log(e.message)
        alert(e.message)
    });
});










//인증 이벤트 처리
btnGoogle.onclick = () => auth.signInWithPopup(provider);

signOutBtn.onclick = () => {auth.signOut(); window.location.reload(true);}
signOutBtn_li.onclick = () => {auth.signOut(); window.location.reload(true);}

auth.onAuthStateChanged(user => {
    if (user) {
        //sign in
        var user = firebase.auth().currentUser;
        var name, email, photoUrl, uid, emailVerified;
        name = user.displayName;
        email = user.email;
        photoUrl = user.photoURL;
        emailVerified = user.emailVerified;
        uid = user.uid;
        whenSignedIn.hidden = false;
        whenSignedOut.hidden = true;
        container1.hidden = true;
        userDetails.innerHTML = email + '<span> 안녕하세요.<span>'

        $("#login_popup").hide();
        $("#logout_li").show();


        
        
        if (self.name != 'reload') {
            self.name = 'reload';
            self.location.reload(true);
        }
        else self.name = ''; 

        var db = firebase.firestore();

        const docRef = db.collection("setting").doc(email);
        
        
        const inputTextField = document.querySelector("#Command_text");
        const saveButton = document.querySelector("#saveButton");
        
        //Command_text save
        saveButton.addEventListener("click", function() {
            const textToSave = inputTextField.value;
            if (!textToSave){
                alert("명령어를 입력하세요");
            } else {
            
            alert('"' + textToSave + '"(으)로 명령어 설정이 완료되었습니다.');
            console.log("Your command :" + textToSave);
            
            docRef.set({
                Command_text : textToSave,
                Udate_Time : Date()
            }, {merge : true}).then(function(){
                console.log("Status saved!");
            }).catch(function(error){
                console.log("Got an error: " + error);
            });

            }
        });
        

        // Add Enter Key Command Submit
        inputTextField.addEventListener("keyup", function(event) {
            if (event.keyCode === 13) {
                saveButton.click();
            }
        });
        

        //Command_voice save
                
        var voiceUrl;
        var files = [];
        var reader ;
        var cmd_text;
        docRef.onSnapshot((doc) => {
            cmd_text = doc.data()["Command_text"];
        })

        document.getElementById("select").onclick = function(e){
            var input = document.createElement('input');
            input.type = 'file';
            input.accept = "audio/*"

            input.onchange = e => {
                files = e.target.files;
                reader = new FileReader();
                reader.readAsDataURL(files[0]);
            }
            input.click();
        }


        document.getElementById('upload').onclick = function(){
            if (files[0]){
                var metadata = {
                    contentType: 'audio/*',
                };
                var uploadTask = firebase.storage().ref('Voices/'+email+'/'+cmd_text + "_" + files[0].name).put(files[0], metadata); // firestore에서 다운로드가 되고
                // var uploadTask = firebase.storage().ref('Voices/'+email+'/'+cmd_text + "_" + files[0].name).put(files[0]); // firestore에서 링크로 음성 연결된다

                uploadTask.on('state_changed', function(snapshot) {
                    var progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
                    console.log(progress+'%');
                    // document.getElementById('UpProgress').innerHTML = 'Upload' + progress + '%';
                },

                function(error) {
                    alert(error);
                    console.log('error :' + error);
                },

                function() {
                    uploadTask.snapshot.ref.getDownloadURL().then(function(url){
                        voiceUrl = url;
                            
                    docRef.set({
                        Command_Voice_Link : voiceUrl,
                        Command_Voice_Name : cmd_text + "_" + files[0].name,
                        Udate_Time : Date()
                    }, {merge : true});
                    });
                    alert('업로드 하였습니다.');
                }
                );
            }
            else{
                alert('파일을 선택해주세요.')
            }
        }


        document.getElementById('cmd_yes').onclick = function(){
            Command_text_btn = $('#cmd').text();
            console.log("Your Command : " + Command_text_btn);
            
            docRef.set({
                Command_text : Command_text_btn,
                Update_Time : Date()
            }, {merge : true});
            alert('"' + Command_text_btn + '"로 명령어 설정이 완료되었습니다');
        }

        
        // // Wifi Setting
        // const create = document.querySelector("#create");
        
        // //Command_text save

        // create.addEventListener("click", function() {
        //     const Wifi_ID = document.getElementById('Wifi_ID').value;
        //     const Wifi_PW = document.getElementById('Wifi_PW').value;

        //     if (!Wifi_ID || !Wifi_PW){
        //         alert("값을 입력하세요");
        //     } else {
        //         var dic = "Wifi ID : " + Wifi_ID + "\nWifi PW : " + Wifi_PW +"\nEmail : " + email;
        //         fileName = "Configure File"
        //         content = dic;
        //         console.log(content);
        //         var blob = new Blob([content], { type: 'text/plain' });
        //         objURL = window.URL.createObjectURL(blob);
                        
        //         // 이전에 생성된 메모리 해제
        //         if (window.__Xr_objURL_forCreatingFile__) {
        //             window.URL.revokeObjectURL(window.__Xr_objURL_forCreatingFile__);
        //         }
        //         window.__Xr_objURL_forCreatingFile__ = objURL;
        //         var a = document.createElement('a');
        //         a.download = fileName;
        //         a.href = objURL;
        //         a.click();
        //     }
        // });

        // // Add Enter Key Command Submit
        // Wifi_PW.addEventListener("keyup", function(event) {
        //     if (event.keyCode === 13) {
        //         create.click();
        //     }
        // });

        // Set Mode
        const mode_normal = document.querySelector("#mode_normal");
        mode_normal.addEventListener("click", function() {
            alert('Normal Mode로 설정되었습니다.');
            console.log("Your Mode : normal");
            docRef.set({
                Mode : "normal",
                Update_Time : Date()
            }, {merge : true}).then(function(){
                console.log("Status saved!");
            }).catch(function(error){
                console.log("Got an error: " + error);
            });
        });

        const mode_security = document.querySelector("#mode_security");
        mode_security.addEventListener("click", function() {
            alert('Security Mode로 설정되었습니다.');
            console.log("Your Mode : security");
            docRef.set({
                Mode : "security",
                Update_Time : Date()
            }, {merge : true}).then(function(){
                console.log("Status saved!");
            }).catch(function(error){
                console.log("Got an error: " + error);
            });
        });

        docRef.onSnapshot((doc) => {
            document.getElementById("current_mode").innerHTML='Current Mode : ' + doc.data()["Mode"];
        })



    } else{
        //sign out
        whenSignedIn.hidden = true;
        whenSignedOut.hidden = false;
        container1.hidden = false;
        userDetails.innerHTML = '';

        $("#login_popup").show();
        $("#logout_li").hide();
  

        const setting_section = document.querySelector("#setting_section");          
         
        setting_section.addEventListener("click", function() {
            alert("Login please")
        });

        const mode_section = document.querySelector("#mode_section");          
         
        mode_section.addEventListener("click", function() {
            alert("Login please")
        });

        // Wifi Setting
        const create = document.querySelector("#create");          
         
        create.addEventListener("click", function() {
            alert("Login please")
        });
    
    }
});


