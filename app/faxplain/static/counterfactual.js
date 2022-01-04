const app = Vue.createApp({
    data() {
        return {
            orig_doc: '',
            explains: '',
            query: '',
            pred: '',
            label: '',
            response: ''
        }
    },
    methods: {
        forward_sentence(query, sentence, label) {
            $.ajax({
                url: "/show_example",
                type: "post",
                contentType: "application/json;charset=UTF-8",
                data: JSON.stringify({
                    query: query,
                    doc: sentence,
                    label: label,
                    use_custom_mask: false,
                    position_scoring_method: 'gradient',
                    word_scoring_method: 'gradient',
                    gramma: false
                    }),
                success: function(response) {
                    this.response = response
                }
            });
        }
    },
    // created: async function(){;
    //     var apiEndpoint = window.location.href
    //     const gResponse = await fetch(apiEndpoint + '-init')
    //     const gObject = await gResponse.json();
    //     this.orig_doc = gObject.orig_doc;
    //     this.query = gObject.query;
    //     this.explains = gObject.explains;
    //     this.pred = gObject.pred;
    //     this.label = gObject.label;
    // }
});
// $(".sentence").click(function() {
//     let sent = $(this).text()
//     $.ajax({
//         url: "/show_example",
//         type: "post",
//         contentType: "application/json;charset=UTF-8",
//         data: JSON.stringify({
//             query: "{{ query }}",
//             doc: sent,
//             label: "{{ label }}",}),
//         success: function(response) {
//                 $("#example").html(response);
//                 $("#user-choice").css('display', "inherit");
//                 $("#query-history .user-chosen-content").remove();
//             },
//         error: function(xhr) {
//             //Do Something to handle error
//         }
//     });
// });

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