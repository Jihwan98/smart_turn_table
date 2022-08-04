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

console.log(firebase)

var provider = new firebase.auth.GoogleAuthProvider();

const auth = firebase.auth();



// function saveToFile_Chrome(fileName, content) {
//     const Wifi_ID = document.getElementById('Wifi_ID').value;
//     const Wifi_PW = document.getElementById('Wifi_PW').value;
    
    
//     var dic = "Wifi ID : " + Wifi_ID + "\nWifi PW : " + Wifi_PW;
    
//     fileName = "Configure File"
//     content = dic;
//     console.log(content);

//     var blob = new Blob([content], { type: 'text/plain' });
//     objURL = window.URL.createObjectURL(blob);
            
//     // 이전에 생성된 메모리 해제
//     if (window.__Xr_objURL_forCreatingFile__) {
//         window.URL.revokeObjectURL(window.__Xr_objURL_forCreatingFile__);
//     }
//     window.__Xr_objURL_forCreatingFile__ = objURL;
//     var a = document.createElement('a');
//     a.download = fileName;
//     a.href = objURL;
//     a.click();
// }



auth.onAuthStateChanged(user => {
    if (user) {
        //sign in
        var user = firebase.auth().currentUser;
        var email
        email = user.email;

        const create = document.querySelector("#create");
        
        //Command_text save
        create.addEventListener("click", function() {
            const Wifi_ID = document.getElementById('Wifi_ID').value;
            const Wifi_PW = document.getElementById('Wifi_PW').value;

            if (!Wifi_ID || !Wifi_PW){
                alert("값을 입력하세요");
            } else {
                var dic = "Wifi ID : " + Wifi_ID + "\nWifi PW : " + Wifi_PW +"\nEmail : " + email;
                fileName = "Configure File"
                content = dic;
                console.log(content);
                var blob = new Blob([content], { type: 'text/plain' });
                objURL = window.URL.createObjectURL(blob);
                        
                // 이전에 생성된 메모리 해제
                if (window.__Xr_objURL_forCreatingFile__) {
                    window.URL.revokeObjectURL(window.__Xr_objURL_forCreatingFile__);
                }
                window.__Xr_objURL_forCreatingFile__ = objURL;
                var a = document.createElement('a');
                a.download = fileName;
                a.href = objURL;
                a.click();
            }
        });

        // Add Enter Key Command Submit
        Wifi_PW.addEventListener("keyup", function(event) {
            if (event.keyCode === 13) {
                create.click();
            }
        });




    } else{
        const create = document.querySelector("#create");          
         
        create.addEventListener("click", function() {
            alert("Login please")
        });
    }
});

