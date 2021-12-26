var app = new Vue({
    data: {

    }
    
})
$(".sentence").click(function() {
    let sent = $(this).text()
    $.ajax({
        url: "/show_example",
        type: "post",
        contentType: "application/json;charset=UTF-8",
        data: JSON.stringify({
            query: "{{ query }}",
            doc: sent,
            label: "{{ label }}",}),
        success: function(response) {
                $("#example").html(response);
                $("#user-choice").css('display', "inherit");
                $("#query-history .user-chosen-content").remove();
            },
        error: function(xhr) {
            //Do Something to handle error
        }
    });
});

$("#apply-mask").click(function() {
    let masked_raw_html = $('#selected-sentence').html();
    let mask_method = $('#masks').val();
    let attr_method = $('#attrs').val();
    let gen_method = $('#new_words').val();
    let gramma_res = $('#gramma_res').is(":checked")
    let ins = $('#ins').is(":checked")
    let del = $('#del').is(":checked")

    console.log(attr_method);
    $.ajax({
        url: "/doc_history",
        type: "post",
        contentType: "application/json;charset=UTF-8",
        data: JSON.stringify({
            query: "{{ query }}",
            masked_raw_html: masked_raw_html,
            label: "{{ label }}",
            mask_method: mask_method,
            attr_method: attr_method,
            gen_method: gen_method,
            gramma_res: gramma_res,
            ins: ins,
            del: del
        }),
        success: function(response) {
                $("#query-history").html(response);
            },
        error: function(xhr) {
            //Do Something to handle error
        }
    });
});