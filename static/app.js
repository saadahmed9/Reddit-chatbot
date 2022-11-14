class Chatbox {
    constructor() {
        this.args = {
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
        }

        
        this.state = false;
        this.messages = []; //To store our messages
    }

    display() {
        const { chatBox, sendButton, topics } = this.args;

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        // topics.addEventListener('click', () => console.log(topics));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    onx(x, y, z, color1) {
        let d1 = {
            type: 'scatter3d',
            mode: 'dots',
            x: [0, x],
            y: [0, y],
            z: [0, z],
            opacity: 1,
            line: {
                width: 6,
                color: color1,
                reversescale: false
            }
        }
        return d1;
    };


    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        var field2 = document.getElementById('field2');
        var x;
        var m1 = []
        var s = ''
        let topic = document.getElementById('topics').value;
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1, topic: topic }
        this.messages.push(msg1);

        fetch('http://0.0.0.0:5000' + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1, topic: topic }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(data => {
                x = data;
                console.log(x)
                let msg2 = { name: "BotMaster", message: data.answer };
                this.messages.push(msg2);
                console.log(this.messages);
                this.updateChatText(chatbox);
                textField.value = '';
                let qn = x.query_embed;
                console.log("qn", qn);
                m1.push(this.onx(qn[0], qn[1], qn[2], 'red'));
                let embed = x.cos_embed;
                console.log()
                s = '<p><p style="color:#FF0000">' + "Queries:  " + x.question + '</p>' + "<p style='color:#005b96'>" + "Answers:  " + x.sim_msg.join("<br>") + '</p></p>';
                embed.forEach(item => {
                    m1.push(this.onx(item[0], item[1], item[2], 'blue'));
                });

                Plotly.newPlot('myDiv', m1, {
                    height: 500,
                    width: 500
                });
                field2.innerHTML = s;

            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox)
                textField.value = ''
            });


        // [{
        //     type: 'scatter3d',
        //     mode: 'dots',
        //     x: [0, 0],
        //     y: [0, 1],
        //     z: [0, 2],
        //     opacity: 1,
        //     line: {
        //         width: 6,
        //         color: 'red',
        //         reversescale: false
        //     }
        // }, {
        //     type: 'scatter3d',
        //     mode: 'dots',
        //     x: [0, 3],
        //     y: [0, 4],
        //     z: [0, 5],
        //     opacity: 1,
        //     line: {
        //         width: 6,
        //         color: 'blue',
        //         reversescale: false
        //     }
        // }]

    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function (item, index) {
            if (item.name === "BotMaster") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;

    }
}


const chatbox = new Chatbox();
chatbox.display();
