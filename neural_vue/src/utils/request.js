import axios from 'axios'


const request = axios.create({ //也就是在输入的url前面加上/api
    baseURL: "http://120.24.230.237:5000",
    timeout: 5000
})



export default request