var JSON_DATA
var IMAGE
var REC_DATA
var ELE = $(".scrollmenu").children();
function setResultData(data) {

    h = $('.bar')
    h[0].setAttribute("data-percent", String(data['happy']*100)+"%");
    h[1].setAttribute("data-percent", String(data['sad']*100)+"%");
    h[2].setAttribute("data-percent", String(data['surprise']*100)+"%");
    h[3].setAttribute("data-percent", String(data['fear']*100)+"%");
    h[4].setAttribute("data-percent", String(data['angry']*100)+"%");
    h[5].setAttribute("data-percent", String(data['neutral']*100)+"%");
    h[6].setAttribute("data-percent", String(data['disgust'] * 100) + "%");
    console.log("Result Updated!")
}

function show_data() {
      setTimeout(function start (){

                            $('.bar').each(function(i){
                              var $bar = $(this);

                              if (cv == 0) {
                                $(this).append('<span class="count col align-self-end"></span>');
                              }

                              setTimeout(function(){
                                $bar.css('width',$bar.attr('data-percent'));      //$bar.attr('data-percent')
                              }, i*100);
                            });
                          cv = 1;
                          $('.count').each(function () {
                              $(this).prop('Counter',0).animate({
                                  Counter: $(this).parent('.bar').attr('data-percent')
                              }, {
                                  duration: 2000,
                                  easing: 'swing',
                                  step: function (now) {
                                      $(this).text(Math.ceil(now) +'%');
                                  }
                              });
                          });

                }, 500)
      }



var cv = 0;
$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();


    function readURL(input) {
        if (input.files && input.files[0]) {

        const fileName = input.files[0].name;
        const reader = new FileReader();
        reader.readAsDataURL(input.files[0]);

          reader.onload = function (e) {
            var img = new Image();
            img.src = e.target.result;
            img.onload = () => {
                const elem = document.createElement('canvas');
                const width = Math.min(200, img.width);
                const scaleFactor = width / img.width;
                elem.width = width;
                elem.height = img.height * scaleFactor;

                const ctx = elem.getContext('2d');
                // img.width and img.height will contain the original dimensions
                ctx.drawImage(img, 0, 0, width, img.height * scaleFactor);
                ctx.canvas.toBlob((blob) => {
                        IMAGE = new File([blob], fileName, {
                        type: 'image/jpeg',
                        lastModified: Date.now()
                    });
                }, 'image/jpeg', 1);
          },
          reader.onerror = error => console.log(error);



                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').hide();
        $("h3").html("your predictions will show below");
        readURL(this);
    });





    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData();

        form_data.append('file',IMAGE);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
          success: function (data) {
            REC_DATA = data
            if (REC_DATA[-1]) {
              $('.loader').hide();
              $('.image-section').hide();
              $("h3").html("No Faces Detected ! please choose another image")
              console.log("no data");
            }
            else {
              JSON_DATA = JSON.parse(JSON.stringify(JSON.parse(data)))
              $(".scrollmenu").html("")
              var len = 0;
              for (i in JSON_DATA) {
                $(".scrollmenu").append("<img src=" + JSON_DATA[i]["face"] + " >");
                len += 1;
              }
              $("h3").html("Here are your Results !");
              if (len == 1) {
                $(".scrollmenu").hide();
                $("h4").html("Here are your Emotions");
              }
              else {
                $("h4").html(String(len) + " Faces Detected");
                $(".scrollmenu").hide();
                $(".scrollmenu").fadeIn(650);
              }


              ELE = $(".scrollmenu").children();
              ELE.click(function () {
                $(this).parent().children().css('border-color', 'yellow');
                $(this).parent().children().css('border-width', '2px');
                $(this).css('border-color', 'tomato');
                $(this).css('border-width', '4px');
                console.log(JSON_DATA[$(this).index()]["predictions"]);
                setResultData(JSON_DATA[$(this).index()]["predictions"]);
                show_data()
              });
              // Get and display the result
              $('.loader').hide();

              $('#result').fadeIn(600);

              setResultData(JSON_DATA[0]["predictions"]);
              show_data()



              console.log('Success!');
            }
          },
        });
    });

});

