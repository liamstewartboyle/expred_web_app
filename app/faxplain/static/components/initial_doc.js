const initial_doc = app.component('initial-doc', {
    props: {
        query: {
            type: String,
            required: true
        },
        orig_doc: {
            type: Array,
            required: true
        },
        explains: {
            type: Array,
            required: true
        },
        label: {
            type: String,
            required: true
        },
        pred:{
            type: String,
            required: true
        },
        pos_img_url: {
            type: String,
            required: true
        },
        neg_img_url: {
            type: String,
            required: true
        }
    },
    data() {
        return {
            show_full_doc: false,
            img_dir: {
                'POS': this.pos_img_url,
                "NEG": this.neg_img_url
            }
        }
    },
    template:
    /*html*/
    `
    <div class='card mb-1'>
        <div class="card-header">
            1. Select a sentence
        </div>
        <div class='card-body'>
            <div class='row'>
                <div class='h5 col-md-auto'>query: </div>
                <div class='h5 col'>{{ query }}</div>
            </div>
            <div class='row'>
                <div class='h5 col-md-auto mt-1 mb-2'>pred: </div>
                <img class='col-sm-auto mt-2 mb-2 img-fluid' :src='img_dir[this.pred]'/>
                <div class='h5 col-md-auto mt-1 mb-2'>label: </div>
                <img class='col-sm-auto mt-2 mb-2 img-fluid' :src='img_dir[this.label]'/>
            </div>

            <hr/>

            <div class='row'>
                <div class='col'>
                    <div
                        :style=" {'display': (show_full_doc ? 'block' : 'none')} "
                        class='full-doc'>
                        <span 
                            v-for="(s, i) in this.orig_doc"
                            class='sentence'
                            v-on:click='select_sentence(s)'>
                            <template v-for="(token, j) in s">
                            <span :class=" (explains[i][j] === 1) ? 'fw-bold' : 'fw-normal' ">
                                {{ token }}
                            </span>{{ ' ' }}
                            </template>
                        </span>
                    </div>
                    <div 
                        :style=" {'display': (show_full_doc ? 'none' : 'block')}"
                        class='doc-rat'>
                        <template v-for='(s, i) in this.orig_doc'>
                            <template v-if='len(explains[i])>0'>
                                <span
                                    class='sentence'
                                    v-on:click='select_sentence(s)'>
                                    <template v-for='(token, j) in s'>
                                        <span :class=" (explains[i][j] === 1) ? 'fw-bold' : 'fw-noral' ">
                                            {{ token }}
                                        </span>{{ ' ' }}
                                    </template>
                                </span>
                            </template>
                            <template v-else>
                                <span>.</span>{{ ' ' }}
                            </template>
                        </template>
                    </div>
                </div>
            </div>
            <div class='row'>
                <div class='col mt-2'>
                    <button 
                        type='button'
                        class='btn btn-secondary'
                        v-on:click='toggle_show()'
                        id='toggle_show_button'>
                        Expand all
                    </button>
                </div>
            </div>
        </div>
    </div>
    `,
    methods: {
        select_sentence(sentence) {
            evt = {
                query: this.query,
                sentence: sentence,
                label:this.label
            }
            this.eventBus.emit('select-sentence', evt)
        },
        len(exp) {
           let total = [];
           exp.forEach(element => {
               total.push(element)
           }); 
           return total.reduce(function(total, num){return total + num}, 0)
        },
        toggle_show(){
            this.show_full_doc = !this.show_full_doc
            if (this.show_full_doc) {
                $('#toggle_show_button').text('Show Rationals Only')
            } else {
                $('#toggle_show_button').text('Show all')
            }
        }
    }
})