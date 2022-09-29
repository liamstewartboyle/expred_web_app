const app = Vue.createApp({
    data() {
        return {
            orig_doc: '',
            explains: '',
            query: '',
            pred: '',
            label: '',
            session_id: undefined,
            cf_examples: [],
            is_cf_example_loading: false,
            is_cf_example_ready: false,
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
            selection_strategy = 'hotflip'
            if (evt.hasOwnProperty('selection_strategy')) {
                selection_strategy = evt['selection_strategy']
            }
            let data = {
                ann_id: evt['ann_id'],
                query: evt['query'],
                doc: evt['sentence'],
                label: evt['label'],
                session_id: this.session_id,
                use_custom_mask: use_custom_mask,
                custom_mask: custom_mask,
                masking_method: masking_method,
                selection_strategy: selection_strategy,
                gramma: false
            }

            let url = "/show_example"
            axios.post(url, data)
                .then(response => {
                    this.cf_examples = response['data']['cf_examples']
                    this.session_id = response['data']['session_id']
                    this.is_cf_example_ready = true
                    this.is_cf_example_loading = false
                })
                .catch(error => {
                    console.log(error)
                })
        });
        this.eventBus.on('evaluation_done', (data) => {
            this.is_cf_example_loading = true
            this.is_cf_example_ready = false
            let url = "/reg_eval"
            data.session_id = this.session_id
            data.cf_examples = this.cf_examples
            axios.post(url, data)
                .then(response => {
                    this.is_cf_example_loading = false
                    this.is_cf_example_ready = false
                })
                .catch(error => {
                    console.log(error)
                })
        });
        this.eventBus.on('alt_word_selected', (evt) => {
            this.is_cf_example_loading = true
            this.is_cf_example_ready = false
            let url = "/select_alt_word";
            let selection_strategy = 'hotflip'
            if (evt.hasOwnProperty('selection_strategy')) {
                selection_strategy = evt['selection_strategy']
            }
            data = {
                query: this.query,
                session_id: this.session_id,
                cf_examples: this.cf_examples,
                alt_word_id: evt['alt_word_id'],
                masking_method: 'custom',
                use_custom_mask: true,
                selection_strategy: selection_strategy,
            }
            axios.post(url, data)
                .then(response => {
                    this.cf_examples = response['data']['cf_examples']
                    this.session_id = response['data']['session_id']
                    this.is_cf_example_ready = true
                    this.is_cf_example_loading = false
                })
                .catch(error => {
                    console.log(error)
                })
        })
    },

    methods: {

    },
});
