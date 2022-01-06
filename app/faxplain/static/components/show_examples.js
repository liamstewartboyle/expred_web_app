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
            plausibility: 3,
            clearance: 3,
            masking_strategy: 'expred',
            try_others: false,
            satisfy: false
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
                mask: mask
            }
            this.eventBus.emit('select-sentence', evt)
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
                plausibility: this.plausibility,
                clearance: this.clearance
            }
            this.eventBus.emit('evaluation_done', data)
        }
    },
    template:
    /*html*/
    `
    <a class='display-7' data-bs-toggle='collapse' href='#user-guidance'>
        User Guidance...
    </a>
    <div class='card card-body container-fluid collapse' id='user-guidance'>
        <div class='mb-1'>
            Place holder for the user guidance.
        </div>
    </div>
    <div class='card container-fluid'>
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
                        <span 
                        v-bind:class='[{ "fw-bold": mask[j] === 1 }, { "mark": cf_example["replaced"] === j }]'>
                        {{ word }}
                        </span>{{' '}}
                    </span>
                    <span>
                        <img :src="img_url[cf_example['pred']]" alt="{{ cf_example['pred'] }}">
                    </span>
                </span>
            </div>

            <hr/>

            <div class='row'>
                <button
                    class='btn btn-success col ms-3 me-3'
                    data-bs-toggle='collapse'
                    href='#evaluate'
                    @click='this.satisfy =! this.satisfy'>
                    Looks nice, evaluate them
                    <span v-if='this.satisfy'>
                        <<<
                    </span>
                    <span v-else>
                        >>>
                    </span>
                </button>
            
                <button 
                    class='btn btn-danger col ms-3 me-3'
                    data-bs-toggle='collapse' 
                    href='#try-others'
                    @click='this.try_others=!this.try_others'> 
                    Looks lame, try another strategy
                    <span v-if='this.try_others'>
                        <<<
                    </span>
                    <span v-else>
                        >>>
                    </span>
                </button>
            </div>
            
            <div class='row'>
                <div class='collapse mt-3 col' id='evaluate'>
                    <div class='card card-body'>
                        <div class='row mb-3'>
                            <div class=col>
                                Evaluation:
                            </div>
                        </div>
                        <div class='row mb-1'>
                            <div class='col'>
                                <label for="plausibility" class="form-label col">Plausibility: ({{plausibility}}/5)</label>
                            </div>
                            <div class='col'>
                                <input v-model='plausibility' type="range" class="form-range" min="1" max="5" step="1" id="plausibility"> 
                            </div>
                            <div class='col'>
                                <label for="clearance" class="form-label">Clearance: ({{clearance}}/5)</label>
                            </div>
                            <div class='col'>
                                <input v-model='clearance' type="range" class="form-range" min="1" max="5" step="1" id="clearance"> 
                            </div>
                        </div>
                        <div class='row mt-4 float-end'>
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

                <div class='collapse mt-3 col' id='try-others'>
                    <div class='card card-body'>
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
                                        v-bind:class='[{ "fw-bold": custom_mask[j] === 1 }, { "mark": examples[0]["replaced"] === j }]'>
                                        {{ word }}
                                        </span>{{' '}}
                                    </span>
                                </span>
                            </div>
                        </div>
                        <div class='row mb-1'>
                            <div class='col'>
                                <label for="attrs">Attribution Strategy:</label>
                            </div>
                            <div class='col'>
                                <select class='form-select' name="attrs" id="attrs">
                                    <optgroup label="Gradient-based">
                                        <option value="gradient">Gradient Attribution (HotFlip)</option>
                                    </optgroup>
                                    <optgroup label="Hierarchical">
                                        <option value="hierarchical">Hierarchical importance</option>
                                    </optgroup>
                                    <optgroup label="Attention-based">
                                        <option value="att">Attention score</option>
                                    </optgroup>
                                </select>
                            </div>
                        </div>
                        <div class='row mb-1'>
                            <div class='col'>
                                <label for="new_words">Replacement Retrieving Strategy:</label>
                            </div>
                            <div class='col'>
                                <select class='form-select' name="new_words" id="new_words">
                                    <optgroup label="Gradient-based">
                                        <option value="gradient">Gradient Attribution (HotFlip)</option>
                                    </optgroup>
                                    <optgroup label="Masked-model-based">
                                        <option value="ptuning">P-Tuning</option>
                                        <option value="t5">T5 model</option>
                                    </optgroup>
                                </select>
                            </div>
                        </div>
                        <div class='row mb-1 mt-4'>
                            <div class='col'>
                                <input class='form-check-input me-1' type="checkbox" id="gramma_res" name="gramma_res">
                                <label for="gramma_res"> Grammatical restriction </label>
                            </div>
                            <div class='col'>
                                <input class='form-check-input me-1' type="checkbox" id="ins" name="ins">
                                <label for="ins">Insertion</label>
                            </div>
                            <div class='col'>
                                <input class='form-check-input me-1' type="checkbox" id="del" name="del">
                                <label for="del">Deletion</label>
                            </div>
                        </div>
                        <div class='row mt-4 float-end'>
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
    `,
})