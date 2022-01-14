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
    mounted() {
        this.eventBus.on('select-sentence', (evt) => {
            this.is_cf_example_loading = true
            this.is_cf_example_ready = false
            use_custom_mask = false
            custom_mask = undefined
            masking_method = 'expred'
            if (evt.hasOwnProperty('mask') && !(typeof evt['mask'] === "undefined")) {
                use_custom_mask = true
                custom_mask = evt['mask']
                masking_method = 'custom'
            }
            let data = {
                query: evt['query'],
                doc: evt['sentence'],
                label: evt['label'],
                use_custom_mask: use_custom_mask,
                custom_mask: custom_mask,
                masking_method: masking_method,
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
        this.eventBus.on('evaluation_done', (data) => {
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
    methods: {},
});
