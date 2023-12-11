let boardImage

setInterval(                               //Periodically
  function()
  {

    fetch("update_images", {
        method: "GET",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    })
        .then(response => response.json().then(json => showBoard(json)))
        .catch(error => alert(`Error: ${error}`))
  },
  2000);

function showBoard(image) {
  if (image.status) {
    document.getElementById("board").src = "data:image/png;base64," + image.data
  }

}
