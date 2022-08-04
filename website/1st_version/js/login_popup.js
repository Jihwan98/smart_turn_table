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
        
        
        self.close();





    } else{
        //sign out
        whenSignedIn.hidden = true;
        whenSignedOut.hidden = false;
        container1.hidden = false;
        userDetails.innerHTML = '';

}
});




