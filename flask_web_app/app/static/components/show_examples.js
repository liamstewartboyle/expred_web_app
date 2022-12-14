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
            type: Object,
            required: true
        },
        query: {
            type: String,
            required: true
        },
        label: {
            type: String,
            required: true
        }
    },
    data() {
        return {
            img_url: {
                'POS': this.pos_img_url,
                'NEG': this.neg_img_url
            },
            mask: this.cf_examples['mask'],
            custom_mask: JSON.parse(JSON.stringify(this.cf_examples['mask'])),
            examples: this.cf_examples['instances'],
            ann_id: this.cf_examples['ann_id'],
            plausibility: 3,
            meaningfulness: 3,
            risk: 3,
            masking_strategy: 'expred',
            selection_strategy: 'hotflip',
            try_others: false,
            satisfy: false,
        }
    },
    methods: {
        toggle_custom_mask(position)  {
            this.custom_mask[position] = 1 - this.custom_mask[position]
        },
        apply_mask() {
            if (this.masking_strategy === 'expred') {
                mask = undefined
            } else {
                mask = this.custom_mask
            }
            evt = {
                query: this.query,
                sentence: this.examples[0]['input'],
                label: this.label,
                mask: mask,
                selection_strategy: this.selection_strategy,
                ann_id: this.ann_id
            }
            this.eventBus.emit('select-sentence', evt)
            this.eventBus.emit('ret-strategy',this.selection_strategy)
        },
        toggle_try_others() {
            this.try_others = !this.try_others
            this.satisfy = false
        },
        toggle_evaluate(){
            this.satisfy =! this.satisfy;
            this.try_others = false
        },
        submit_eval() {
            data = {
                eval: {
                    plausibility: this.plausibility,
                    meaningfulness: this.meaningfulness,
                    risk: this.risk
                }
            }
            this.eventBus.emit('evaluation_done', data)
        },
        select_word(event) {
            data = {
                alt_word_id: event.currentTarget.id,
                selection_strategy: this.selection_strategy
            }
            this.eventBus.emit('alt_word_selected', data)
        }
    },
    template:
    /*html*/
    `
    <div class='mb-1'>
        <a class='display-7' data-bs-toggle='collapse' href='#user-guidance'>
            User Guidance...
        </a>
        <div class='card card-body container-fluid collapse mb-1' id='user-guidance'>
                Place holder for the user guidance.
        </div>
    </div>
    <div class='card'>
        <div class="card-header">
            2. How do you like the counterfactuals?
        </div>
        <div class='card-body'>
            <div class='h5 row'>
                <div class='col-md-auto'>
                    Example with:
                </div>
            </div>
            <div v-for='(cf_example, i) in examples' class='row'>
                <span class=''>- {{ i }} word<span v-if='(i > 1)'>s</span> swapped: </span>
                <span class="rec-counter" id="selected-sentence">
                    <span v-for='(word, j) in cf_example["input"]'>
                        <span :class="{ 'fw-bold': (mask[j] === 1) }">
                            <span v-if="cf_example['replaced'] === j" class="badge rounded-pill bg-warning text-dark">
                                <span v-if="cf_example['alternative_words'] === undefined">
                                    {{ word }}
                                </span>
                                <span v-else>
                                    <span class="dropdown">
                                        <a class=" dropdown-toggle" data-bs-toggle="dropdown" 
                                           href="#" role="button" aria-expanded="false">
                                           {{ word }}
                                        </a>
                                        <ul class="dropdown-menu">
                                            <li v-for='(alt_word, k) in cf_example["alternative_words"]'>
                                                <a class="dropdown-item" 
                                                    v-bind:id="i+ '.' + j + '.' + k" 
                                                    @click="select_word($event)">
                                                    {{ alt_word }}
                                                </a>
                                            </li>
                                        </ul>
                                    </span>
                                </span>
                            </span>
                            <span v-else>
                                {{ word }}
                            </span>
                        </span>{{' '}}
                    </span>
                    <span>
                        <img :src="img_url[cf_example['pred']]" alt="{{ cf_example['pred'] }}">
                    </span>
                </span>
            </div>

            <hr/>

            <div class='accordion' id='feedbackAccordion'>
                <div class='accordion-item'>
                    <h2 class='accordion-header' id='headingEval'>
                        <button 
                            @click='this.satisfy =! this.satisfy'
                            class='accordion-button collapsed'
                            type='button' 
                            data-bs-toggle='collapse'
                            data-bs-target='#evaluation'
                            aria-expanded="true"
                            aria-controls="evaluation">
                            Looks nice, evaluate them
                        </button>
                    </h2>

                    <div 
                        id='evaluation'
                        class='accordion-collapse collapse'
                        area-labelledby='headingEval'
                        data-bs-parent='#feedbackAccordion'>
                        <div class='accordion-body'>                        
                            <div class='row mb-1'>
                                <div class='col'>
                                    <label for="plausibility" class="form-label col">Plausibility: ({{plausibility}}/5)</label>
                                </div>
                                <div class='col'>
                                    <input v-model='plausibility' type="range" class="form-range" min="1" max="5" step="1" id="plausibility"> 
                                </div>
                                <div class='col'>
                                    <label for="meaningfulness" class="form-label">Meaningfulness: ({{meaningfulness}}/5)</label>
                                </div>
                                <div class='col'>
                                    <input v-model='meaningfulness' type="range" class="form-range" min="1" max="5" step="1" id="meaningfulness"> 
                                </div>
                            </div>
                            <div class='row mb-1'>
                                <div class='col'>
                                    <label for="risk" class="form-label col">Risk of the model: ({{risk}}/5)</label>
                                </div>
                                <div class='col'>
                                    <input v-model='risk' type="range" class="form-range" min="1" max="5" step="1" id="risk"> 
                                </div>
                            </div>
                            <div class='row mt-4'>
                                <div class='col'>
                                    <button 
                                    class='btn btn-primary'
                                    id="submit-eval"
                                    type="button"
                                    name="submit-eval"
                                    @click='submit_eval()'>Submit evaluation</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class='accordion-item'>
                    <h2 class='accordion-header' id='headingTryOthers'>
                        <button 
                            class='accordion-button collapsed'
                            type='button'
                            data-bs-toggle='collapse'
                            data-bs-target='#try-others'
                            aria-expanded='false'
                            aria-controls="try-others">
                            Looks lame, try another strategy
                        </button>
                    </h2>
                    <div 
                        id='try-others'
                        class='accordion-collapse collapse'
                        area-labelledby='headingTryOthers'
                        data-bs-parent='#feedbackAccordion'>
                        <div class='accordion-body'>
                            <div class='row mb-1'>
                                <div class='col'>
                                    <label for="masks">Masking strategy:</label>
                                </div>
                                <div class='col'>
                                    <select class='form-select' v-model='masking_strategy' name="masks" id="masks">
                                        <option value="expred">ExPred</option>
                                        <option value="custom">Custom</option>
                                    </select>
                                </div>
                            </div>
                            <div class='row card card-body mb-3 ms-1 me-1' v-show='masking_strategy === "custom"'>
                                <div class='col-sm-auto'>
                                    <div class='row'>Custom Masking:</div>
                                </div>
                                <div clas='col'>
                                    <span class="rec-counter" id="selected-sentence">
                                        <span v-for='(word, j) in examples[0]["input"]'>
                                            <span 
                                            @click='toggle_custom_mask(j)'
                                            v-bind:class='[{ "fw-bold": custom_mask[j] === 1 }, ]'>
                                            {{ word }}
                                            </span>{{' '}}
                                        </span>
                                    </span>
                                </div>
                            </div>
                            <div class='row mb-1'>
                                <div class='col'>
                                    <label for="new_words">Replacement Retrieving Strategy:</label>
                                </div>
                                <div class='col'>
                                    <select class='form-select' v-model='selection_strategy' name="new_words" id="new_words">
                                        <optgroup label="Gradient-based">
                                            <option value="hotflip">Taylor-expansion (HotFlip)</option>
                                        </optgroup>
                                        <optgroup label="Masked-model-based">
                                            <option value="mlm">Masked Language Model</option>
                                        </optgroup>
                                    </select>
                                </div>
                            </div>
                            <div class='row mt-4'>
                                <div class='col'>
                                    <button 
                                    class='btn btn-primary'
                                    id="apply-mask"
                                    type="button"
                                    name="predict-mask"
                                    @click='apply_mask()'>Try this</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>      
    </div>
    `,
})