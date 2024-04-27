$('#spinner').hide();

$('#predict-btn').on('click', function () {
    console.log('sending request to /api/v1/predict...')

    var fileInput = $('#image-input').prop('files')[0];
    var form = new FormData();
    form.append('image', fileInput);
    form.append('num_classes', $('#num-class').val());

    $.ajax({
        url: '/api/v1/predict/form-data',
        type: 'POST',
        enctype: 'multipart/form-data',
        data: form,
        contentType: false,
        processData: false,
        beforeSend: function () {
            $("#spinner").show();
            $('#result-display').hide();
        },
        success: function (data) {
            console.log(data);
            $('#result-display').show();
            $("#spinner").hide();

            $('#display-total-time').text(data['total_time']);
            $('#display-predictions').empty();
            $('#display-predictions').text('Kết quả: ')


            for (var i = 0; i < data['predicts'].length; i++) {
                var prediction = data['predicts'][i];
                var predictionDiv = '<span class="badge text-bg-success">' + prediction['class_name'] + ':' + prediction['confidence'] + '</span>';
                $('#display-predictions').append(predictionDiv);
            }

            var editor = new JsonEditor('#json-display', data);
        }
    });
});