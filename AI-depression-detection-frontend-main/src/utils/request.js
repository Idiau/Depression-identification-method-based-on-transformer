import axios from 'axios'
const service = axios.create({
    baseURL: "http://127.0.0.1:5000" ,
    timeout: 5000 // request timeout
})



export default service