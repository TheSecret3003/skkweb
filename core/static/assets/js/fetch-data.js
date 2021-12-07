window.addEventListener("load", function() {
    $('#submit').click(function (e) { 
        e.preventDefault();
        if ($('#FIELD_NAME').value != '' 
            && $('#FIELD_IOIP').value != '' 
            && $('#RECOVERY_FACTOR').value != '' )
        load_profile()    
    });
});


function load_profile() {
    var xhttp = new XMLHttpRequest();

    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var response = JSON.parse(this.response);
            var data1 = response.data1;
            var data2 = response.data2;

            var profile_production = {
                x: Object.values(data1["X_up"]),
                y: Object.values(data1["Y_up"]),
                line: {color: "dimgrey", width: 2},
                fill: 'tozeroy',
                fillcolor: 'red'
            };

            var profile_production1 = {
                x: Object.values(data2["X_down"]),
                y: Object.values(data2["Y_down"]),
                line: {color: "dimgrey", width: 2},
                
            };

            var plot_data = [profile_production,profile_production1];

            var plot_layout = {
                title:response.fieldname,
                titlecolor: 'dimgrey',
                titlefont: { size:30 },
                width: 600,
                height: 400,
                margin: {
                    l: 50,
                    r: 50,
                    b: 50,
                    t: 50,
                    pad: 4
                },
                xaxis: {
                    title: 'Time, Month',
                    titlefont: { size:15 },
                    titlecolor: 'dimgrey',
                    titlefontstyle: 'italic'
                },
                yaxis:{
                    title: 'Oil Production, STB/M',
                    titlefont: { size:15 },
                    titlecolor: 'dimgrey',
                    titlefontstyle: 'italic'
                }
            };

            var plot_config = {responsive: true, displayModeBar: false};

            Plotly.newPlot(document.getElementById("profile_chart"), plot_data, plot_layout, plot_config);
        }
    };
    var endpoint = "profile"
    var params = {
        FIELD_NAME: document.getElementById("FIELD_NAME").value,
        FIELD_IOIP: document.getElementById("FIELD_IOIP").value,
        RECOVERY_FACTOR: document.getElementById("RECOVERY_FACTOR").value

    }
    var url = endpoint + formatParams(params)
    xhttp.open("GET", url);
    xhttp.send();
}

function formatParams( params ){
    return "?" + Object
          .keys(params)
          .map(function(key){
            return key+"="+encodeURIComponent(params[key])
          })
          .join("&")
  }