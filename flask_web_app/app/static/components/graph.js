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
        }
    },


    template:
     /*html*/
        `
        <div class="chart-container" style="position: relative; height:100%; width:100%">
             <canvas id="myChart"></canvas>
        </div>
        

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
                if(newData.data.y+1>this.config.options.scales.y.max){
                    this.config.options.scales.y.max=newData.data.y+1
                }
                if(type===0){
                        this.chartData.datasets[0].data.push(newData.data)
                }else{
                        this.chartData.datasets[1].data.push(newData.data)
                }
                if(newData.prev !== null){
                    this.chartData.datasets.push(this.generateLine(newData.prev.data.y,newData.prev.data.x,newData.data.x))
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
                    indexAxis:'y',
                    borderColor: 'rgb(75, 192, 192)',
                    order:2}
                return line;
            },
        generateData(){
                class chart_object{
                    constructor(prev,type,data,pos){
                        this.prev=prev;
                        this.type = type;
                        this.data={x:data,y:pos,r:15};
                    }
                }
                let arr=[]
                let prev = null
                for(let i=0;i<this.examples.length;i++){
                    if(i===0){
                        let c = new chart_object(null,0,Number(this.examples[i]['score']),i)
                        arr.push(c)
                        prev = c
                    }else{
                        let c = new chart_object(prev,0,Number(this.examples[i]['score']),i)
                        arr.push(c)
                        prev = c
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
                    let altIndex = this.findAlternativeWord()
                    for(let i = altIndex;i<this.dataArray[n].length;i++){
                        if(i === altIndex){
                            this.addData(this.dataArray[n][i],1)
                        }else{
                            this.addData(this.dataArray[n][i],0)
                        }
                    }
                }
            },
        findAlternativeWord(){
                let n = this.dataArray.length
                let l = 0
                let obj1 = null
                let obj2 = null
                if(this.dataArray[n-2].length>=this.dataArray[n-1].length){
                    l = this.dataArray[n-1].length
                }else{
                    l = this.dataArray[n-2].length
                }

                for(let i = 0;i<l;i++){
                    obj1 = this.dataArray[n-2][i]
                    obj2 = this.dataArray[n-1][i]
                    if(obj1.data.x !== obj2.data.x){
                            return i
                    }

                }
                return 0
            },
        resetData(newData){
               if(this.examples !== null){
                   if(Number(this.examples[0]['score'])!== Number(newData[0]['score'])){
                       this.dataArray = []
                       this.chartData.datasets = [
                                          {
                                          label: 'Bubble Computer',
                                          data: [],
                                          type:'bubble',
                                          backgroundColor: 'rgb(255,99,132)',
                                            order:1
                                      }, {
                                          label: 'Bubble Human',
                                          data: [],
                                          type: 'bubble',
                                          backgroundColor: 'rgb(0,0,255)',
                                          order:1
                                        }]
                       this.config.options.scales.y.max = 1
                   }
               }
        }




    },

    mounted(){

          this.chartData = {
              datasets:[
                  {
                  label: 'Bubble Computer',
                  data: [],
                  type:'bubble',
                  backgroundColor: 'rgb(255,99,132)',
                    order:1
              }, {
                  label: 'Bubble Human',
                  data: [],
                  type: 'bubble',
                  backgroundColor: 'rgb(0,0,255)',
                  order:1
                }],
              yLabels: Array.from({length:100},(item,index)=>index),
              clip: {left:false, top: 10, right: false, bottom: false}
            }

            this.config = {
              type: 'line',
              data: this.chartData,
              options: {
                  maintainAspectRatio:false,
                layout:{
                  padding:10
                  },
                scales:{
                  x:{
                      type:"linear",
                      min:0,
                      max:1,
                      ticks:{
                          stepSize:0.1
                    }
                  },
                  y:{
                      type:'category',
                      //position: 'right',
                      min:-0.5,
                      max:0,
                      ticks:{
                          stepSize:1
                    }
                  }
                },
                responsive:true,
                  plugins:{
                    legend:{
                        labels:{
                            filter: function(item,chart){
                                return !item.text.includes('line')
                            }
                        }
                    }
                  }
              }

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

        this.myChart.update()



    }


})
