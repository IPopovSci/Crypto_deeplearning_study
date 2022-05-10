
      $(document).ready(function() {

        $('#all_intervals').change(function(){

          $.getJSON('/_update_dropdown', {
                                      selected_ticker: $('#all_tickers').val(),
                selected_interval: $('#all_intervals').val(),
            selected_model_type: $('#all_model_types').val()


          }).success(function(data) {
                $('#all_tickers').html(data.ticker_selection);
                $('#all_model_types').html(data.model_type_selection);
                $('#all_model').html(data.model_selection);
                })

        $("option[value='Select Interval']").remove();
        });

        $('#all_tickers').change(function(){


          $.getJSON('/_update_dropdown', {
                          selected_ticker: $('#all_tickers').val(),
                selected_interval: $('#all_intervals').val(),
            selected_model_type: $('#all_model_types').val()

          }).success(function(data) { console.log('Im a maniac????')
                $('#all_model_types').html(data.model_type_selection);
                                $('#all_model').html(data.model_selection);
           })

        $("option[value='Select Ticker']").remove();
        });

        $('#all_model_types').change(function(){

          $.getJSON('/_update_dropdown', {
                          selected_ticker: $('#all_tickers').val(),
                selected_interval: $('#all_intervals').val(),
            selected_model_type: $('#all_model_types').val()

          }).success(function(data) { console.log('Im a maniac????')
                $('#all_model').html(data.model_selection);
           })
          $("option[value='Select Model Type']").remove();
        });

          $('#process_input').bind('click', function() {
          $.ajax({
            url: '/_process_data',
            dataType: 'json',
            data: $('form').serialize(),
            type: 'POST',
            success: function(response) {
                console.log(response['preds'])
                $('#preds').text(response['preds']).show();
            }})
			return false;
          });
});

