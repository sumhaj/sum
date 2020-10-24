//const result=document.getElementById("pass");
//const generate=document.getElementById("pass_btn");

document.getElementById("pass_btn").addEventListener('click',() => {
document.getElementById("pass").value = generatePassword(10);
});

function generatePassword(length){
	let generatedPassword='';

	for(let i=0; i<length; i+=4){
     
        var l=getRandomLower();
	    var u=getRandomUpper();
	    var n=getRandomNumber();
	    var s=getRandomSymbol();
        const typesArr=[l,u,n,s];
        typesArr.forEach(type => {
        	generatedPassword+=type;
     });
	}
	const finalPassword = generatedPassword.slice(0, length);
	return finalPassword;

}


function getRandomLower(){

return String.fromCharCode(Math.floor(Math.random()*26)+97);
}

function getRandomUpper(){

return String.fromCharCode(Math.floor(Math.random()*26)+65);
}

function getRandomNumber(){

return String.fromCharCode(Math.floor(Math.random()*10)+48);
}

function getRandomSymbol(){
const symbols='!@#$%^&*(){}[]=<>/,.';
return symbols[Math.floor(Math.random()*symbols.length)];
}

// function formatState (state) {
//   if (!state.id) {
//     return state.text;
//   }
//   var baseUrl = "/user/pages/images/flags";
//   var $state = $(
// 	`<span> ${state.text}</span>`
//   );
//   return $state;
// };

$(".dropdown1").select2({
  
});
$(".dropdown2").select2({
  
});

document.getElementById("email_btn").addEventListener('click',checkEmail);

function checkEmail(){
	var inputUser=document.getElementById("email").value;
 	fetch('data.json')
 	.then(res =>res.json())
	.then(data => {
		data.forEach(users =>{
	 	var user=users.email;
	 	if(inputUser==user){
			document.getElementById("hidden").style.display='block';
			document.getElementById("space").style.display='none';
	 	}
		 });
	});
}		 	

document.getElementById("submit_btn").addEventListener('click',(e) => {
	e.preventDefault();
	
	let grp='';
	let data=$(".dropdown1").find(":selected").text();
	for(let i=0;i<data.length;i++){
    	if(i==0){
			grp+=data[i];
		}
		else{
			grp+=data[i]+',';
		}
	}

let grp1='';
	let dat=$(".dropdown2").val();
	for(let j=0;j<dat.length;j++){
    	if(j==0){
			grp1+=dat[j];
		}
		else{
		grp1+=', '+dat[j];
		}
	}

	document.getElementById('display').innerHTML='Email Address: '+document.form.email_name.value
	+'<br></br>'+'Password: '+document.form.pass_name.value+'<br></br>'+'Groups: '+grp+'<br></br>'+'Plants ID: '+grp1;
});