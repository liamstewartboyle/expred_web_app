var mouseDown = false;
    var currentLabelId = -1
    function toggle_background(element) {
        var cbid = element.htmlFor
        var cb = document.getElementById(cbid)
        if (cb.checked) {
            element.style.background = '#FFFF00'
        } else {
            element.style.background = 'white'
        }
    }

    function customToggleCheckBox(l){
        var cbid = l.htmlFor;
        currentLabelId = cbid
        var cb = document.getElementById(cbid);
        cb.checked = !cb.checked;
    }

    $("label").on('click', function(event){
        event.preventDefault();
    });

    $("label").on('mousedown touchstart', function(event) {
        event.preventDefault();
        mouseDown = true;
        customToggleCheckBox(this)
        toggle_background(this)
    });

    $("label").on('mousemove touchmove', function(event) {
        event.preventDefault();
        var cbid = this.htmlFor;
        if (currentLabelId !== cbid && mouseDown) {
            customToggleCheckBox(this)
            toggle_background(this)
        }
    });
    $(window.document).on('mouseup touchend', function(event) {
        // Capture this event anywhere in the document,
        // since the mouse may leave our element while
        // mouse is down and then the 'up' event will
        // not fire within the element.
        mouseDown = false;
        currentLabelId = -1
    });

    window.onload = function() {
        var all_ps = document.getElementsByTagName("select");
        for (var i=0, max=all_ps.length; i < max; i++) {
            let cls_select = document.getElementById('cls' + parseInt(i));
            var selected;
            if (cls_select.classList.contains('REFUTES')) {
                selected = 'REFUTES'
            } else {
                selected = 'SUPPORTS'
            }
            cls_select.value = selected;
        }

        var all_labels = document.getElementsByTagName("label");
        for (var i=0, max=all_labels.length; i < max; i++) {
            var element = all_labels[i];
            var cbid = element.htmlFor;
            var cb = document.getElementById(cbid);
            if (cb.classList.contains("1") || cb.classList.contains('1.0')) {
                cb.checked = true;
            }
            if (cb.checked) {
                element.style.background = '#FFFF00'
            } else {
                element.style.background = 'white'
            }
        }
    };

$(".show-more a").on("click", function() {
    var $this = $(this);
    var $content = $this.parent().prev("div.content");
    var linkText = $this.text().toUpperCase();

    if(linkText === "SHOW MORE..."){
        linkText = "Fold";
        $content.switchClass("hideContent", "showContent", 400);
    } else {
        linkText = "Show more...";
        $content.switchClass("showContent", "hideContent", 400);
    };

    $this.text(linkText);
});