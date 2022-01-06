const app = Vue.createApp({
    data() {
        return {
            orig_doc: '',
            explains: '',
            query: '',
            pred: '',
            label: '',
            cf_examples: [],
            is_cf_example_loading: false,
            is_cf_example_ready: false,
            is_annotation_done: false
        }
    },
    mounted(){
        this.eventBus.on('select-sentence', (evt) => {
            this.is_cf_example_loading = true
            this.is_cf_example_ready = false
            use_custom_mask = false
            custom_mask = undefined
            if (evt.hasOwnProperty('mask') && !(typeof evt['mask'] === "undefined") ) {
                use_custom_mask = true
                custom_mask = evt['mask']
            }
            let data = {
                query: evt['query'],
                doc: evt['sentence'],
                label: evt['label'],
                use_custom_mask: use_custom_mask,
                custom_mask: custom_mask,
                position_scoring_method: 'gradient',
                word_scoring_method: 'gradient',
                gramma: false
            }
            let url = "/show_example"
            axios.post(url, data)
                .then(response => {
                    this.cf_examples = response['data']['cf_examples']
                    this.is_cf_example_ready = true
                    this.is_cf_example_loading = false
                })
                .catch(error => {
                    console.log(error)
                })
        });
        this.eventBus.on('evaluation_done', (data) =>{
            let url = "/reg_eval"
            axios.post(url, data)
                .then(response => {
                    this.is_cf_example_loading = false
                    this.is_cf_example_ready = false
                    this.is_annotation_done = true
                })
                .catch(error => {
                    console.log(error)
                })
        })
    },
    methods: {
    },
});

// $("#apply-mask").click(function() {
//     let masked_raw_html = $('#selected-sentence').html();
//     let mask_method = $('#masks').val();
//     let attr_method = $('#attrs').val();
//     let gen_method = $('#new_words').val();
//     let gramma_res = $('#gramma_res').is(":checked")
//     let ins = $('#ins').is(":checked")
//     let del = $('#del').is(":checked")

//     console.log(attr_method);
//     $.ajax({
//         url: "/doc_history",
//         type: "post",
//         contentType: "application/json;charset=UTF-8",
//         data: JSON.stringify({
//             query: "{{ query }}",
//             masked_raw_html: masked_raw_html,
//             label: "{{ label }}",
//             mask_method: mask_method,
//             attr_method: attr_method,
//             gen_method: gen_method,
//             gramma_res: gramma_res,
//             ins: ins,
//             del: del
//         }),
//         success: function(response) {
//                 $("#query-history").html(response);
//             },
//         error: function(xhr) {
//             //Do Something to handle error
//         }
//     });
// });