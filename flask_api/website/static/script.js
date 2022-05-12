
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
                $('#preds1h').text(response['preds1h']).show();
                $('#preds4h').text(response['preds4h']).show();
                $('#preds12h').text(response['preds12h']).show();
                $('#preds24h').text(response['preds24h']).show();
                $('#preds48h').text(response['preds48h']).show();
                $('#sc1h').text(response['sc1h']).show();
                $('#sc4h').text(response['sc4h']).show();
                $('#sc12h').text(response['sc12h']).show();
                $('#sc24h').text(response['sc24h']).show();
                $('#sc48h').text(response['sc48h']).show();
                document.getElementById('image_backtest').src = 'data:;base64,' + response['image_backtest'];
                document.getElementById('image_graph').src = 'data:;base64,' + response['image_graph'];
            }})
			return false;
          });
});

