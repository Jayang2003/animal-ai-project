import React, { useState } from "react";
import "../App.css";
import { FaEye, FaEyeSlash } from "react-icons/fa";
import { MdEmail } from "react-icons/md";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const API = "http://127.0.0.1:8000";

const Login = () => {
  const [form, setForm] = useState({
    email: "",
    password: ""
  });

  const [show, setShow] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
  };

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!form.email || !form.password) {
      alert("Enter all fields");
      return;
    }

    try {
      const res = await axios.post(`${API}/login`, {
        username: form.email,
        password: form.password
      });

      if (res.data.success) {
        alert("Login Success 🚀");
        navigate("/dashboard");
      } else {
        alert("Invalid credentials ❌");
      }
    } catch (err) {
      console.error(err);
      alert("Server error ❌");
    }
  };

  return (
    <div className="container">
      <form className="login-box" onSubmit={handleLogin}>
        <h2>Login</h2>

        <label>Email</label>
        <div className="input-group">
          <MdEmail className="icon" />
          <input
            type="text"
            name="email"
            placeholder="Enter email"
            value={form.email}
            onChange={handleChange}
          />
        </div>

        <label>Password</label>
        <div className="input-group">
          <input
            type={show ? "text" : "password"}
            name="password"
            placeholder="Enter password"
            value={form.password}
            onChange={handleChange}
          />

          <span className="eye" onClick={() => setShow(!show)}>
            {show ? <FaEyeSlash /> : <FaEye />}
          </span>
        </div>

        <button type="submit">Login</button>
      </form>
    </div>
  );
};

export default Login;