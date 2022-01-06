const show_examples = app.component('show-examples', {
    props: {
        pos_img_url: {
            type: String,
            required: true
        },
        neg_img_url: {
            type: String,
            required: true
        },
        cf_examples: {
            type: Array,
            required: true
        },
    },
    data() {
        return {
            img_url: {
                'POS': this.pos_img_url,
                'NEG': this.neg_img_url
            },
            mask: cf_examples['mask']
        }
    },
    mounted (){
        // console.log('aaa')
        // this.eventBus.on('select-sentence', (evt) => {
        //     let data = {
        //         query: evt['query'],
        //         doc: evt['sentence'],
        //         label: evt['label'],
        //         use_custom_mask: false,
        //         position_scoring_method: 'gradient',
        //         word_scoring_method: 'gradient',
        //         gramma: false
        //     }
        //     let url = "/show_example"
        //     axios.post(url, data)
        //         .then(response => {
        //             this.cf_exmples = response['data']['cf_example']
        //             this.eventBus.emit('cf-example-ready', 1)
        //         })
        //         .catch(error => {
        //             console.log(error)
        //         })
        // })
    },
    methods: {
    },
    template:
    /*html*/
    `
    <div v-for='(cf_example, i) in cf_examples["instances"]' class='row'>
        <span class='fw-bold'>- {{ i }} word<span v-if='(i > 1)'>s</span> swapped: </span>
        <span class="rec-counter" id="selected-sentence">
            <span v-for='(word, j) in cf_example["input"]'>
                <span v-bind:class='[{ "fw-bold": cf_examples["mask"][j] === 1 }, { "mark": cf_example["replaced"] === j }]'>{{ word }}</span>{{' '}}
            </span>
            <span>
            <img :src="img_url[cf_example['pred']]" alt="{{ cf_example['pred'] }}">
        </span>
    </div>
    `,
})