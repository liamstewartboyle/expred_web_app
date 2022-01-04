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
            show_full_doc: false
        }
    },
    template:
    /*html*/
    `
    <div class='container'>
    <div class='row'>
        <div class='col-sm-5'>
            <div class='row'>
                <div class='h3'>Query: <div class='fst-italic'>{{ query }}</div></div>
            </div>
            <button 
                type='button'
                class='btn btn-secondary p-2 mb-2 row'
                v-on:click='toggle_show()'
                id='toggle_show_button'>
                Show all
            </button>
            <div
                :style=" {'display': (show_full_doc ? 'block' : 'none')} "
                class='full-doc row'>
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
                class='doc-rat row'>
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
    </div>
    `,
    methods: {
        select_sentence(sentence) {
            this.$emit('select-sentence', this.query, sentence, this.label)
        },
        get_img_src(label) {
            if (label === 'POS') {
                return this.pos_img_url
            }
            return this.neg_img_url
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