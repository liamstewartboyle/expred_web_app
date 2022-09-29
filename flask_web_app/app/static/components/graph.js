const graph = app.component('graph', {

    props:{
        cf_examples: {
            type: Object,
            required: true
        }
    },

    data(){
        return{
            chartData:null,
            config:null,
            examples: this.cf_examples['instances'],
        }
    },


    template:
     /*html*/
        `
        <div class="chart-container" style="position: relative; height:80vh; width:40vw">
             <canvas id="myChart"></canvas>
        </div>

    `,
    mounted(){


          this.chartData = {
              datasets:[
                  {
                  label: 'Bubble Computer',
                  data: [],
                  type:'bubble',
                  backgroundColor: 'rgb(255,99,132)',
                    order:2
              }, {
                  label: 'Bubble Human',
                  data: [],
                  type: 'bubble',
                  backgroundColor: 'rgb(0,0,255)',
                  order:2
                }],
              labels: Array.from({length:100},(item,index)=>index),
              clip: {left:false, top: 10, right: false, bottom: false}
            }

            this.config = {
              type: 'bubble',
              data: this.chartData,
              options: {
                layout:{
                  padding:10
                  },
                scales:{
                  x:{
                    min:0,
                    ticks:{
                      stepSize:1
                    }
                  },
                  y:{
                    position: 'right',
                    min:0,
                    max:1,
                    ticks:{
                      stepSize:0.1
                    }
                  }
                },
                responsive:true,
              }
            }



    },

    updated(){
            console.log(this.examples)

          const myChart = new Chart(
              document.getElementById('myChart'),
              this.config,
          );
    }


})
