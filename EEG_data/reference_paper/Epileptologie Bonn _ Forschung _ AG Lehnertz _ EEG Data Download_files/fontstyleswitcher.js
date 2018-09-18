function fontsizeup() {
  active = getActiveFontStyleSheet();
  switch (active) {
    case 'A' : 
      setActiveFontStyleSheet('A+');
      break;
    case 'A+' : 
      setActiveFontStyleSheet('A++');
      break;
    case 'A++' :
      break;
    default :
      setActiveFontStyleSheet('A');
      break;
  }
}

function fontsizedown() {
  active = getActiveFontStyleSheet();
  switch (active) {
    case 'A++' : 
      setActiveFontStyleSheet('A+');
      break;
    case 'A+' : 
      setActiveFontStyleSheet('A');
      break;
    case 'A' : 
      break;
    default :
      setActiveFontStyleSheet('A');
      break;
  }
}

function setActiveFontStyleSheet(title) {
  var i, a, main;
  for(i=0; (a = document.getElementsByTagName("link")[i]); i++) {
    if(a.getAttribute("rel").indexOf("style") != -1 && a.getAttribute("title")) {
      a.disabled = true;
      if(a.getAttribute("title") == title) a.disabled = false;
    }
  }
}

function getActiveFontStyleSheet() {
  var i, a;
  for(i=0; (a = document.getElementsByTagName("link")[i]); i++) {
    if(a.getAttribute("rel").indexOf("style") != -1 && a.getAttribute("title") && !a.disabled) return a.getAttribute("title");
  }
  return null;
}

function getPreferredFontStyleSheet() {
  return ('A');
}

function createFontCookie(name,value,days) {
  if (days) {
    var date = new Date();
    date.setTime(date.getTime()+(days*24*60*60*1000));
    var expires = "; expires="+date.toGMTString();
  }
  else expires = "";
  document.cookie = name+"="+value+expires+"; path=/";
}

function readFontCookie(name) {
  var nameEQ = name + "=";
  var ca = document.cookie.split(';');
  for(var i=0;i < ca.length;i++) {
    var c = ca[i];
    while (c.charAt(0)==' ') c = c.substring(1,c.length);
    if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
  }
  return null;
}

window.onload = function(e) {
  var cookie = readFontCookie("style");
  var title = cookie ? cookie : getPreferredFontStyleSheet();
  setActiveFontStyleSheet(title);
}

window.onunload = function(e) {
  var title = getActiveFontStyleSheet();
  createFontCookie("style", title, 365);
}

var cookie = readFontCookie("style");
var title = cookie ? cookie : getPreferredFontStyleSheet();
if (title == 'null') {
  title = getPreferredFontStyleSheet();
}

setActiveFontStyleSheet(title);