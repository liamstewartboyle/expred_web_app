export function get_img_src(label) {
    if (label === 'POS') {
        return url_for('static', filename='imgs/POS.png')
    }
    return url_for('static', filename='imgs/NEG.png')
}