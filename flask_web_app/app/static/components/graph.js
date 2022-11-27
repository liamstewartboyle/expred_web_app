class chart_object{
    constructor(prev,type,data,pos,text){
        this.prev=prev;
        this.type = type;
        this.data={x:data,y:pos,r:15};
        this.text = text;
    }
}
let textList = []

function getSentence(y) {
    let text;
    for(let i = 0; i<textList.length;i++){
        if(textList[i][0] === y){
            text = textList[i][1]
        }
    }
    document.getElementById("text").innerHTML = text.join(" ");

}

const graph = app.component('graph', {
    props:{
        cf_examples: {
            type: Object,
            required: true
        }
    },

    data(){
        this.myChart = null
        return{
            chartData:null,
            config:null,
            dataArray:[],
            init:0,
            examples:null,
            selection_strategy:'hotflip',
            flag:0,
            alt_word_i: 0,
        }
    },


    template:
     /*html*/
        `

        <div class="chart-container" style="position: relative; height:50vh; width:100%">
             <canvas id="myChart"></canvas>
        </div>
        <p id="text"></p>
        

    `,

    methods:{

            initChart(){
                if(this.init ===0) {
                    this.init = 1;
                    this.myChart = new Chart(
                    document.getElementById('myChart'),
                    this.config,)
                }
            },
            addData(newData,type){


                //dynamic scales
                if(newData.data.x+1>this.config.options.scales.x.max){
                    this.config.options.scales.x.max=newData.data.x+1
                }


                if(type===0){
                    if(this.selection_strategy==='hotflip'){
                        this.chartData.datasets[0].data.push(newData.data)
                    }else{
                        this.chartData.datasets[2].data.push(newData.data)
                    }
                }else{
                        this.chartData.datasets[1].data.push(newData.data)
                }
                if(newData.prev !== null){
                    this.chartData.datasets.push(this.generateLine(newData.prev.data.x,newData.prev.data.y,newData.data.y))
                }
            },
            generateLine(start,prev,cur){
                let lineData=[]
                for(let i=0;i<start;i++){
                    lineData.push(null)
                }
                lineData.push(prev)
                lineData.push(cur)
                const line = {
                    label: 'line',
                    data:lineData,
                    type:'line',
                    //indexAxis:'x',
                    borderColor: 'rgb(75, 192, 192)',
                    order:2}
                return line;
            },
        generateData(){
                let arr=[]
                let prev = null
                for(let i=0;i<this.examples.length;i++){
                    if(i===0){
                        let c = new chart_object(null,0,i,Number(this.examples[i]['score']),this.examples[i]['input'])
                        arr.push(c)
                        prev = c
                        textList.push([Number(this.examples[i]['score']),this.examples[i]['input']])
                    }else{
                        let c = new chart_object(prev,0,i,Number(this.examples[i]['score']),this.examples[i]['input'])
                        arr.push(c)
                        prev = c
                        textList.push([Number(this.examples[i]['score']),this.examples[i]['input']])
                    }

            }
                this.dataArray.push(arr)
        },
        addDataToGraph(){
                let n = this.dataArray.length -1
                if(n<1){
                    for(let i = 0;i<this.dataArray[0].length;i++){
                        this.addData(this.dataArray[0][i],0)
                    }
                }else{
                    let altIndex = this.alt_word_i;
                    for(let i = altIndex;i<this.dataArray[n].length;i++){
                        if(i === altIndex && altIndex !== 0){
                            console.log(this.dataArray[n][i])
                            this.addData(this.dataArray[n][i],1)
                        }else{
                            this.addData(this.dataArray[n][i],0)
                        }
                    }
                    this.alt_word_i = 0;
                }
            }

        ,
        resetData(newData){
               if(this.examples !== null){
                   if(Number(this.examples[0]['score'])!== Number(newData[0]['score'])){
                       this.dataArray = []
                       this.chartData.datasets = [
                                          {
                                          label: 'Hotflip',
                                          data: [],
                                          type:'bubble',
                                          backgroundColor: 'rgb(255,99,132)',
                                            order:1
                                      }, {
                                          label: 'Alternative Word',
                                          data: [],
                                          type: 'bubble',
                                          backgroundColor: 'rgb(0,0,255)',
                                          order:1
                                        },{
                                            label: 'MLM',
                                            data: [],
                                            type: 'bubble',
                                            backgroundColor: 'rgb(124,252,50)',
                                            order: 1
                                        }]
                       this.config.options.scales.x.max = 1

                       this.setMinMax()
                   }
               }
        },
        setMinMax(){
                let min = 0.4
                let max = 0.6
                let datasets = this.chartData.datasets
                for(let i = 0;i<3;i++){
                    for(let j = 0;j<datasets[i].data.length;j++){
                        if(min>datasets[i].data[j].y){
                            min = datasets[i].data[j].y
                        }
                    }
                }
                for(let i = 0;i<3;i++){
                    for(let j = 0;j<datasets[i].data.length;j++){

                        if(max<datasets[i].data[j].y){
                            max = datasets[i].data[j].y
                        }
                    }
                }


                this.config.options.scales.y.max=max+0.1
                this.config.options.scales.y.min=min-0.1


        },




    },

    mounted() {
        this.eventBus.on('ret-strategy', (selection_strategy) => {
            this.selection_strategy = selection_strategy
            this.flag = 1
        })

        this.eventBus.on('alt_word_selected', (alt_word) => {
            this.alt_word_i = alt_word['alt_word_id'].split('.')[0]
        })

        this.chartData = {
            datasets: [
                {
                    label: 'Hotflip',
                    data: [],
                    type: 'bubble',
                    backgroundColor: 'rgb(255,99,132)',
                    order: 1
                }, {
                    label: 'Alternative Word',
                    data: [],
                    type: 'bubble',
                    backgroundColor: 'rgb(0,0,255)',
                    order: 1
                },
                {
                    label: 'MLM',
                    data: [],
                    type: 'bubble',
                    backgroundColor: 'rgb(124,252,50)',
                    order: 1
                }
            ],
            xLabels: Array.from({length: 100}, (item, index) => index),
            clip: {left: false, top: 10, right: false, bottom: false}
        }


        this.config = {
            type: 'bubble',
            data: this.chartData,
            options: {
                maintainAspectRatio: false,
                layout: {
                    padding: 10
                },
                scales: {
                    x: {
                        type: 'category',
                        min: -0.5,
                        max: 0,
                        ticks: {
                            stepSize: 1
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.1
                        }
                    }
                },
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            filter: function (item, chart) {
                                return !item.text.includes('line')
                            }
                        }
                    }
                },
                onClick: function(e){
                    let elements = this.getElementsAtEventForMode(e, 'nearest', {intersect: true}, true);
                    if(elements.length!==0) {
                        getSentence(elements[0]['element']['$context']['raw'].y)
                    }

              }
            },


            }

    },

    updated(){
        //initialize chart
        this.initChart()
        //init examples
        this.resetData(this.cf_examples['instances'])
        this.examples = this.cf_examples['instances']


        this.generateData();

        this.addDataToGraph()
        this.setMinMax()
        this.myChart.update()



    }


})
